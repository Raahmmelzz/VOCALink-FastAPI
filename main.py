from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

from sqlalchemy import (
    create_engine, Column, Integer, String,
    ForeignKey, text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

from pydantic import BaseModel, EmailStr
from typing import Optional
from passlib.context import CryptContext

import jwt
import datetime as dt
import os
import io
import json
import random
import string
import requests

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM   = "HS256"

HF_API_URL  = "https://api-inference.huggingface.co/models/rammealz123/VOCALink-Mobile-STT"
HF_TOKEN    = os.getenv("HUGGINGFACE_TOKEN")
HF_HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vocalink.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine       = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()
pwd_context  = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="VocaLink API")

# ── CORS — explicit list required when allow_credentials=True ───────────────
ALLOWED_ORIGINS = [
    "http://localhost:8081",
    "http://localhost:19006",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:19006",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://vocalink-fastapi.onrender.com",
    # Add your Expo tunnel / ngrok URL here while testing on device
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True)
    email           = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    status          = Column(String, default="STUDENT")  # STUDENT | TEACHER

    teacher_profile   = relationship("TeacherProfile", back_populates="user", uselist=False)
    student_profile   = relationship("StudentProfile",  back_populates="user", uselist=False)
    sent_messages     = relationship("Message", foreign_keys="Message.sender_id",   back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.receiver_id", back_populates="receiver")


class TeacherProfile(Base):
    __tablename__  = "teacher_profiles"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"))
    first_name     = Column(String, default="")
    last_name      = Column(String, default="")
    display_name   = Column(String, default="")
    contact_number = Column(String, default="")
    room_section   = Column(String, default="")
    department     = Column(String, default="")
    grade_handled  = Column(String, default="")
    organization   = Column(String, default="")
    bio            = Column(String, default="")

    user     = relationship("User",           back_populates="teacher_profile")
    students = relationship("StudentProfile", back_populates="instructor")
    session  = relationship("ClassSession",   back_populates="teacher", uselist=False)


class StudentProfile(Base):
    __tablename__   = "student_profiles"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    instructor_id   = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="SET NULL"), nullable=True)
    first_name      = Column(String, nullable=True)
    last_name       = Column(String, nullable=True)
    bio             = Column(String, nullable=True)
    grade_level     = Column(String, nullable=True)
    disability_type = Column(String, nullable=True)

    instructor = relationship("TeacherProfile", back_populates="students")
    user       = relationship("User",           back_populates="student_profile")


# ── DB-backed session — survives Render restarts / spin-downs ───────────────
class ClassSession(Base):
    __tablename__ = "class_sessions"
    id           = Column(Integer, primary_key=True, index=True)
    teacher_id   = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="CASCADE"), unique=True)
    session_code = Column(String)
    is_active    = Column(Boolean, default=True)
    started_at   = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

    teacher = relationship("TeacherProfile", back_populates="session")


# ── Caption messages — saved to DB, polled every 1-2 s by students ──────────
class CCMessage(Base):
    __tablename__ = "cc_messages"
    id         = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="CASCADE"), nullable=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id", ondelete="SET NULL"), nullable=True)
    text       = Column(String)
    speaker    = Column(String, default="teacher")
    sent_at    = Column(String, default=lambda: dt.datetime.utcnow().isoformat())


# ── Direct messages (Messages screen) ───────────────────────────────────────
class Message(Base):
    __tablename__ = "messages"
    id          = Column(Integer, primary_key=True, index=True)
    sender_id   = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    receiver_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    text        = Column(String)
    is_aac      = Column(Boolean, default=False)
    sent_at     = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

    sender   = relationship("User", foreign_keys=[sender_id],   back_populates="sent_messages")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="received_messages")


class AACLog(Base):
    __tablename__ = "aac_logs"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    session_id = Column(Integer, ForeignKey("class_sessions.id", ondelete="SET NULL"), nullable=True)
    icon_id    = Column(String)
    icon_label = Column(String)
    message    = Column(String, nullable=True)
    tapped_at  = Column(String, default=lambda: dt.datetime.utcnow().isoformat())


Base.metadata.create_all(engine)

# ── Safe auto-migrations for existing DBs ────────────────────────────────────
_migrations = [
    ("teacher_profiles", "first_name VARCHAR DEFAULT ''"),
    ("teacher_profiles", "last_name VARCHAR DEFAULT ''"),
    ("teacher_profiles", "grade_handled VARCHAR DEFAULT ''"),
    ("teacher_profiles", "organization VARCHAR DEFAULT ''"),
    ("teacher_profiles", "bio VARCHAR DEFAULT ''"),
    ("cc_messages",      "teacher_id INTEGER"),
    ("class_sessions",   "is_active BOOLEAN DEFAULT 1"),
    ("cc_messages",      "session_id INTEGER"),
    ("aac_logs",         "session_id INTEGER"),
]
for _table, _col in _migrations:
    try:
        with engine.connect() as _conn:
            _conn.execute(text(f"ALTER TABLE {_table} ADD COLUMN {_col}"))
            _conn.commit()
    except Exception:
        pass


# ─────────────────────────────────────────────
# 3. SCHEMAS
# ─────────────────────────────────────────────

class RegisterSchema(BaseModel):
    username: str
    email:    EmailStr
    password: str
    status:   str = "STUDENT"

class LoginSchema(BaseModel):
    identifier: str
    password:   str

class ProfileUpdate(BaseModel):
    first_name:      Optional[str] = None
    last_name:       Optional[str] = None
    bio:             Optional[str] = None
    grade_level:     Optional[str] = None
    disability_type: Optional[str] = None

class ProfileUpdateSchema(BaseModel):
    username:       Optional[str]      = None
    email:          Optional[EmailStr] = None
    first_name:     Optional[str]      = None
    last_name:      Optional[str]      = None
    display_name:   Optional[str]      = None
    contact_number: Optional[str]      = None
    room_section:   Optional[str]      = None
    department:     Optional[str]      = None
    grade_handled:  Optional[str]      = None
    organization:   Optional[str]      = None
    bio:            Optional[str]      = None

class BroadcastSchema(BaseModel):
    text:    str
    speaker: str = "teacher"

class MessageSchema(BaseModel):
    receiver_id: int
    text:        str
    is_aac:      bool = False

class AACLogSchema(BaseModel):
    icon_id:    str
    icon_label: str
    message:    Optional[str] = None

class TTSSchema(BaseModel):
    text: str


# ─────────────────────────────────────────────
# 4. HELPERS & DEPENDENCIES
# ─────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(data: dict) -> str:
    payload = {**data, "exp": dt.datetime.utcnow() + dt.timedelta(days=7)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(auth.split(" ")[1], SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(User).filter(User.id == payload.get("user_id")).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _make_code(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


# ─────────────────────────────────────────────
# 5. AUTH
# ─────────────────────────────────────────────

@app.post("/api/auth/register/")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(
        (User.username == data.username) | (User.email == data.email)
    ).first():
        raise HTTPException(status_code=400, detail="Username or email already taken")

    user = User(
        username        = data.username,
        email           = data.email,
        hashed_password = pwd_context.hash(data.password),
        status          = data.status,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    try:
        if user.status == "TEACHER":
            if not db.query(TeacherProfile).filter_by(user_id=user.id).first():
                db.add(TeacherProfile(user_id=user.id))
        else:
            if not db.query(StudentProfile).filter_by(user_id=user.id).first():
                db.add(StudentProfile(user_id=user.id))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[register] profile warning: {e}")

    return {"message": "User created successfully"}


@app.post("/api/auth/login/")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.username == data.identifier) | (User.email == data.identifier)
    ).first()
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token({"user_id": user.id}), "status": user.status}


# ─────────────────────────────────────────────
# 6. PROFILE
# ─────────────────────────────────────────────

@app.get("/api/profile/me")
def get_profile(current_user: User = Depends(get_current_user)):
    if current_user.status == "TEACHER":
        p = current_user.teacher_profile
        return {
            "id": current_user.id, "username": current_user.username,
            "email": current_user.email, "status": current_user.status,
            "first_name": p.first_name if p else "",
            "last_name":  p.last_name  if p else "",
            "display_name": p.display_name if p else "",
            "department":   p.department   if p else "",
            "grade_handled":p.grade_handled if p else "",
            "room_section": p.room_section  if p else "",
            "bio":          p.bio           if p else "",
        }

    p = current_user.student_profile
    teacher_name, teacher_id = "", None
    if p and p.instructor_id and p.instructor:
        ins = p.instructor
        teacher_name = ins.first_name or ins.display_name or "Teacher"
        teacher_id   = ins.user_id
    return {
        "id": current_user.id, "username": current_user.username,
        "email": current_user.email, "status": current_user.status,
        "first_name":      p.first_name      if p else "",
        "last_name":       p.last_name       if p else "",
        "grade_level":     p.grade_level     if p else "",
        "disability_type": p.disability_type if p else "",
        "bio":             p.bio             if p else "",
        "teacher_name":    teacher_name,
        "teacher_id":      teacher_id,
    }


@app.put("/api/profile/me")
def update_profile(
    data: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    p = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    for field in ("first_name","last_name","bio","grade_level","disability_type"):
        val = getattr(data, field)
        if val is not None:
            setattr(p, field, val)
    db.commit()
    return {"message": "Profile updated"}


@app.patch("/api/users/me/")
def update_me(
    data: ProfileUpdateSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if data.username: current_user.username = data.username
    if data.email:    current_user.email    = data.email
    p = current_user.teacher_profile
    if p:
        for f in ("first_name","last_name","display_name","contact_number",
                  "room_section","department","grade_handled","organization","bio"):
            v = getattr(data, f, None)
            if v is not None:
                setattr(p, f, v)
    db.commit()
    return {"message": "Profile updated"}


@app.get("/api/users/me/")
def get_me(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Web App.tsx fetches this to populate the sidebar teacher name."""
    if current_user.status == "TEACHER":
        p = current_user.teacher_profile
        return {
            "id":           current_user.id,
            "username":     current_user.username,
            "email":        current_user.email,
            "status":       current_user.status,
            "display_name": p.display_name if p else "",
            "first_name":   p.first_name   if p else "",
            "last_name":    p.last_name    if p else "",
        }
    p = current_user.student_profile
    return {
        "id":         current_user.id,
        "username":   current_user.username,
        "email":      current_user.email,
        "status":     current_user.status,
        "first_name": p.first_name if p else "",
        "last_name":  p.last_name  if p else "",
    }


@app.delete("/api/profile/me")
def delete_account(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db.delete(current_user)
    db.commit()
    return {"message": "Account deleted"}


# ─────────────────────────────────────────────
# 7. SESSION  (DB-backed — survives Render restarts)
# ─────────────────────────────────────────────

@app.post("/api/sessions/toggle")
def toggle_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")

    tp = current_user.teacher_profile
    if not tp:
        raise HTTPException(status_code=404, detail="No teacher profile")

    existing = db.query(ClassSession).filter_by(teacher_id=tp.id).first()

    if existing and existing.is_active:
        # END the session
        existing.is_active = False
        db.commit()
        return {"active": False, "session_code": None}
    else:
        # START a new session
        code = _make_code()
        if existing:
            existing.is_active    = True
            existing.session_code = code
            existing.started_at   = dt.datetime.utcnow().isoformat()
        else:
            db.add(ClassSession(teacher_id=tp.id, session_code=code))
        db.commit()
        return {"active": True, "session_code": code}


@app.get("/api/sessions/teacher")
def check_teacher_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Teacher polls this on mount to restore session state after navigation."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return {"active": False, "session_code": None}
    sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
    if sess:
        return {"active": True, "session_code": sess.session_code}
    return {"active": False, "session_code": None}


@app.get("/api/sessions/all/")
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Web LiveCC page: list all teacher sessions for the session selector."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return []
    sessions = (
        db.query(ClassSession)
        .filter_by(teacher_id=tp.id)
        .order_by(ClassSession.started_at.desc())
        .limit(50)
        .all()
    )
    return [
        {"id": s.id, "session_code": s.session_code,
         "started_at": s.started_at, "is_active": s.is_active}
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}/log/")
def get_session_log(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full chronological log for a session: teacher CC + student AAC taps."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        raise HTTPException(status_code=404, detail="No teacher profile")
    sess = db.query(ClassSession).filter_by(id=session_id, teacher_id=tp.id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    cc_msgs = (
        db.query(CCMessage)
        .filter_by(session_id=session_id)
        .order_by(CCMessage.sent_at.asc())
        .all()
    )

    student_ids = [s.user_id for s in tp.students]
    aac_logs = []
    if student_ids:
        aac_logs = (
            db.query(AACLog)
            .filter(AACLog.session_id == session_id, AACLog.user_id.in_(student_ids))
            .order_by(AACLog.tapped_at.asc())
            .all()
        )

    name_map = _build_student_name_map(db, student_ids) if student_ids else {}

    entries = []
    for m in cc_msgs:
        ts = m.sent_at or ""
        entries.append({
            "type":     "cc",
            "id":       f"cc-{m.id}",
            "sort_key": ts,
            "time":     ts[11:16] if len(ts) > 15 else ts,
            "speaker":  "Teacher",
            "text":     m.text,
        })
    for l in aac_logs:
        ts = l.tapped_at or ""
        entries.append({
            "type":     "aac",
            "id":       f"aac-{l.id}",
            "sort_key": ts,
            "time":     ts[11:16] if len(ts) > 15 else ts,
            "speaker":  name_map.get(l.user_id, f"Student #{l.user_id}"),
            "text":     l.message or l.icon_label,
            "icon_id":  l.icon_id,
        })

    entries.sort(key=lambda e: e["sort_key"])

    return {
        "session_id":   sess.id,
        "session_code": sess.session_code,
        "started_at":   sess.started_at,
        "is_active":    sess.is_active,
        "entries":      entries,
    }


@app.get("/api/sessions/student")
def check_student_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Student polls this every 5 s on home screen to detect when class starts."""
    if current_user.status != "STUDENT":
        return {"active": False}
    sp = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
    if not sp or not sp.instructor_id:
        return {"active": False}
    sess = db.query(ClassSession).filter_by(
        teacher_id=sp.instructor_id, is_active=True
    ).first()
    if sess:
        return {"active": True, "session_code": sess.session_code}
    return {"active": False}


# ─────────────────────────────────────────────
# 8. BROADCAST  (teacher types → saved to DB instantly)
# ─────────────────────────────────────────────

@app.post("/api/broadcast/")
def broadcast_to_students(
    data: BroadcastSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")

    tp = current_user.teacher_profile
    sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
    if not sess:
        raise HTTPException(status_code=400, detail="No active session — start class first")

    msg = CCMessage(
        teacher_id = tp.id,
        session_id = sess.id,
        text       = data.text,
        speaker    = data.speaker,
        sent_at    = dt.datetime.utcnow().isoformat(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"id": msg.id, "message": "Broadcasted"}


# ─────────────────────────────────────────────
# 9. CAPTION POLLING  (both teacher feed + student feed)
# Students call this every 1.5 s with ?since=<last_id>
# ─────────────────────────────────────────────

@app.get("/api/cc/messages/")
def get_cc_messages(
    since: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status == "STUDENT":
        sp = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
        tid = sp.instructor_id if sp else None
        if not tid:
            return []
        # Only return messages when the session is actually active
        sess = db.query(ClassSession).filter_by(teacher_id=tid, is_active=True).first()
        if not sess:
            return []
        msgs = (
            db.query(CCMessage)
            .filter(CCMessage.teacher_id == tid, CCMessage.id > since)
            .order_by(CCMessage.id.asc())
            .limit(30)
            .all()
        )
    else:
        # Teacher fetches their own sent captions for the live-room feed
        tp = current_user.teacher_profile
        msgs = (
            db.query(CCMessage)
            .filter(CCMessage.teacher_id == tp.id, CCMessage.id > since)
            .order_by(CCMessage.id.asc())
            .limit(30)
            .all()
        )

    return [
        {"id": m.id, "text": m.text, "speaker": m.speaker, "sent_at": m.sent_at}
        for m in msgs
    ]


# ─────────────────────────────────────────────
# 10. DIRECT MESSAGES
# ─────────────────────────────────────────────

@app.post("/api/messages/")
def send_message(
    data: MessageSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    msg = Message(
        sender_id   = current_user.id,
        receiver_id = data.receiver_id,
        text        = data.text,
        is_aac      = data.is_aac,
        sent_at     = dt.datetime.utcnow().isoformat(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"id": msg.id, "message": "Sent"}


@app.get("/api/messages/my-teacher")
def get_messages_with_teacher(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status != "STUDENT":
        return []
    sp = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
    if not sp or not sp.instructor_id or not sp.instructor:
        return []
    other_id = sp.instructor.user_id
    msgs = (
        db.query(Message)
        .filter(
            ((Message.sender_id == current_user.id) & (Message.receiver_id == other_id)) |
            ((Message.sender_id == other_id) & (Message.receiver_id == current_user.id))
        )
        .order_by(Message.sent_at.asc())
        .all()
    )
    return [
        {"id": m.id, "sender_id": m.sender_id, "receiver_id": m.receiver_id,
         "text": m.text, "is_aac": m.is_aac, "sent_at": m.sent_at}
        for m in msgs
    ]


@app.get("/api/messages/my-students")
def get_messages_from_students(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return []
    ids = [s.user_id for s in tp.students]
    if not ids:
        return []
    msgs = (
        db.query(Message)
        .filter(
            ((Message.sender_id.in_(ids)) & (Message.receiver_id == current_user.id)) |
            ((Message.sender_id == current_user.id) & (Message.receiver_id.in_(ids)))
        )
        .order_by(Message.sent_at.asc())
        .all()
    )
    return [
        {"id": m.id, "sender_id": m.sender_id, "receiver_id": m.receiver_id,
         "text": m.text, "is_aac": m.is_aac, "sent_at": m.sent_at}
        for m in msgs
    ]


# ─────────────────────────────────────────────
# 11. ROSTER MANAGEMENT
# ─────────────────────────────────────────────

@app.get("/api/teacher/students/")
def get_my_students(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        raise HTTPException(status_code=404, detail="Profile not found")
    return [
        {"id": s.user_id, "username": s.user.username,
         "first_name": s.first_name or "", "last_name": s.last_name or ""}
        for s in tp.students
    ]


@app.get("/api/users/all-students/")
def get_all_students(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    return [
        {"id": s.user_id, "username": s.user.username,
         "first_name": s.first_name or "", "last_name": s.last_name or "",
         "assigned": s.instructor_id is not None}
        for s in db.query(StudentProfile).join(User).all()
    ]


@app.post("/api/teacher/students/{user_id}")
def add_student(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    s = db.query(StudentProfile).filter_by(user_id=user_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    s.instructor_id = current_user.teacher_profile.id
    db.commit()
    return {"message": "Student added"}


@app.delete("/api/teacher/students/{user_id}")
def remove_student(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    s = db.query(StudentProfile).filter_by(user_id=user_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    s.instructor_id = None
    db.commit()
    return {"message": "Student removed"}


# ─────────────────────────────────────────────
# 12. AAC LOGS
# ─────────────────────────────────────────────

@app.post("/api/logs/")
def log_icon_tap(data: AACLogSchema, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session_id = None
    if current_user.status == "STUDENT":
        sp = db.query(StudentProfile).filter_by(user_id=current_user.id).first()
        if sp and sp.instructor_id:
            sess = db.query(ClassSession).filter_by(teacher_id=sp.instructor_id, is_active=True).first()
            if sess:
                session_id = sess.id
    db.add(AACLog(
        user_id    = current_user.id,
        session_id = session_id,
        icon_id    = data.icon_id,
        icon_label = data.icon_label,
        message    = data.message,
        tapped_at  = dt.datetime.utcnow().isoformat(),
    ))
    db.commit()
    return {"message": "Log saved"}


@app.get("/api/logs/")
def get_logs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    logs = db.query(AACLog).filter_by(user_id=current_user.id).order_by(AACLog.id.desc()).limit(50).all()
    return [{"id": l.id, "icon_id": l.icon_id, "icon_label": l.icon_label,
             "message": l.message, "tapped_at": l.tapped_at} for l in logs]


def _build_student_name_map(db: Session, student_user_ids: list) -> dict:
    """Return {user_id: display_name} for a list of student user IDs."""
    users    = {u.id: u for u in db.query(User).filter(User.id.in_(student_user_ids)).all()}
    profiles = {sp.user_id: sp for sp in
                db.query(StudentProfile).filter(StudentProfile.user_id.in_(student_user_ids)).all()}
    result = {}
    for uid in student_user_ids:
        sp   = profiles.get(uid)
        u    = users.get(uid)
        name = ((sp.first_name or "") + " " + (sp.last_name or "")).strip() if sp else ""
        result[uid] = name or (u.username if u else f"Student #{uid}")
    return result


def _format_log(l: AACLog, name_map: dict) -> dict:
    return {
        "id":           l.id,
        "user_id":      l.user_id,
        "student_name": name_map.get(l.user_id, f"Student #{l.user_id}"),
        "icon_id":      l.icon_id,
        "icon_label":   l.icon_label,
        "message":      l.message,
        "tapped_at":    l.tapped_at,
    }


@app.get("/api/teacher/logs/")
def get_teacher_logs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Web LiveCC page: full history of every student's AAC taps for this teacher."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return []
    ids = [s.user_id for s in tp.students]
    if not ids:
        return []
    logs = (
        db.query(AACLog)
        .filter(AACLog.user_id.in_(ids))
        .order_by(AACLog.tapped_at.asc())
        .limit(500)
        .all()
    )
    name_map = _build_student_name_map(db, ids)
    return [_format_log(l, name_map) for l in logs]


@app.get("/api/sessions/logs/")
def get_session_logs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Broadcast page sidebar: AAC taps from the current active session only."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    tp = current_user.teacher_profile
    if not tp:
        return []
    sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
    if not sess:
        return []
    ids = [s.user_id for s in tp.students]
    if not ids:
        return []
    logs = (
        db.query(AACLog)
        .filter(
            AACLog.user_id.in_(ids),
            AACLog.tapped_at >= sess.started_at,
        )
        .order_by(AACLog.tapped_at.asc())
        .limit(200)
        .all()
    )
    name_map = _build_student_name_map(db, ids)
    return [_format_log(l, name_map) for l in logs]


# ─────────────────────────────────────────────
# 13. TTS
# ─────────────────────────────────────────────

@app.post("/api/tts/")
def text_to_speech(data: TTSSchema, current_user: User = Depends(get_current_user)):
    try:
        from gtts import gTTS
        tts = gTTS(text=data.text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg",
                                 headers={"Content-Disposition": "inline; filename=tts.mp3"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# ─────────────────────────────────────────────
# 14. STT
# ─────────────────────────────────────────────

@app.post("/api/stt/")
async def speech_to_text(audio: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        audio_bytes = await audio.read()
        resp = requests.post(HF_API_URL, headers=HF_HEADERS, data=audio_bytes, timeout=30)
        out  = resp.json()
        if resp.status_code == 503 or (isinstance(out, dict) and "estimated_time" in out):
            return {"error": "Model warming up", "estimated_time": out.get("estimated_time", 20)}
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=str(out))
        if isinstance(out, list) and out:
            return {"text": out[0].get("text", "")}
        return {"text": out.get("text", str(out))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {e}")