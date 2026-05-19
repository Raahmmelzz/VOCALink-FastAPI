from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, Response
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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
SECRET_KEY    = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM     = "HS256"
SMTP_EMAIL    = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
otp_store: dict = {}

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

# ── CORS — open to all origins (JWT Bearer tokens used, not cookies) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Bulletproof CORS — handles preflight + error responses ───────────────────
@app.middleware("http")
async def force_cors(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


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
    is_verified     = Column(Boolean, default=False)

    teacher_profile = relationship("TeacherProfile", back_populates="user", uselist=False)
    student_profile = relationship("StudentProfile",  back_populates="user", uselist=False)


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
    sessions = relationship("ClassSession",   back_populates="teacher")


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
    last_seen = Column(String, nullable=True)

    instructor = relationship("TeacherProfile", back_populates="students")
    user       = relationship("User",           back_populates="student_profile")


# ── DB-backed session — one row per session, multiple per teacher ────────────
class ClassSession(Base):
    __tablename__ = "class_sessions"
    id           = Column(Integer, primary_key=True, index=True)
    teacher_id   = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="CASCADE"))
    session_code = Column(String)
    is_active    = Column(Boolean, default=True)
    started_at   = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

    teacher = relationship("TeacherProfile", back_populates="sessions")


# ── Caption messages — saved to DB, polled every 1-2 s by students ──────────
class CCMessage(Base):
    __tablename__ = "cc_messages"
    id         = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="CASCADE"), nullable=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id", ondelete="SET NULL"), nullable=True)
    text       = Column(String)
    speaker    = Column(String, default="teacher")
    sent_at    = Column(String, default=lambda: dt.datetime.utcnow().isoformat())




class AACLog(Base):
    __tablename__ = "aac_logs"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    session_id = Column(Integer, ForeignKey("class_sessions.id", ondelete="SET NULL"), nullable=True)
    icon_id    = Column(String)
    icon_label = Column(String)
    message    = Column(String, nullable=True)
    tapped_at  = Column(String, default=lambda: dt.datetime.utcnow().isoformat())


Base.metadata.create_all(bind=engine)

# ── Safe auto-migrations for existing DBs ────────────────────────────────────
_migrations = [
    ("users",            "is_verified BOOLEAN DEFAULT 0"),
    ("student_profiles", "last_seen VARCHAR"),
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

# Drop the unique constraint on class_sessions.teacher_id so each session
# gets its own row.  Needed for existing databases.
try:
    with engine.connect() as _conn:
        if DATABASE_URL.startswith("sqlite"):
            _row = _conn.execute(text(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='class_sessions'"
            )).fetchone()
            if _row and "UNIQUE" in (_row[0] or "").upper():
                _conn.execute(text("""
                    CREATE TABLE class_sessions_v2 (
                        id         INTEGER NOT NULL PRIMARY KEY,
                        teacher_id INTEGER REFERENCES teacher_profiles(id) ON DELETE CASCADE,
                        session_code VARCHAR,
                        is_active  BOOLEAN DEFAULT 1,
                        started_at VARCHAR
                    )
                """))
                _conn.execute(text(
                    "INSERT INTO class_sessions_v2 "
                    "SELECT id, teacher_id, session_code, is_active, started_at "
                    "FROM class_sessions"
                ))
                _conn.execute(text("DROP TABLE class_sessions"))
                _conn.execute(text("ALTER TABLE class_sessions_v2 RENAME TO class_sessions"))
                _conn.commit()
        else:
            _conn.execute(text(
                "ALTER TABLE class_sessions "
                "DROP CONSTRAINT IF EXISTS class_sessions_teacher_id_key"
            ))
            _conn.commit()
except Exception as _e:
    print(f"[migration] class_sessions unique removal: {_e}")


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

class AACLogSchema(BaseModel):
    icon_id:    str
    icon_label: str
    message:    Optional[str] = None

class TTSSchema(BaseModel):
    text: str


class VerifyEmailSchema(BaseModel):
    email: str
    code: str

class SessionLogSchema(BaseModel):
    session_code: str
    icon_id: str
    icon_label: str

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

    # Send verification email
    code = str(random.randint(100000, 999999))
    expires = dt.datetime.utcnow() + dt.timedelta(minutes=30)
    otp_store[f"verify_{user.email}"] = {"otp": code, "expires_at": expires}

    if SMTP_EMAIL and SMTP_PASSWORD:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "VocaLink — Verify Your Email"
            msg["From"]    = SMTP_EMAIL
            msg["To"]      = user.email
            html = f"""
            <div style="font-family:sans-serif;max-width:480px;margin:auto;padding:32px;background:#f9f9f9;border-radius:12px">
              <h2 style="color:#1AADDC">Welcome to VocaLink!</h2>
              <p>Enter this code to verify your email:</p>
              <div style="font-size:36px;font-weight:800;letter-spacing:8px;color:#1A1A2E;padding:16px 0">{code}</div>
              <p style="color:#6B7280;font-size:13px">This code expires in <strong>30 minutes</strong>.</p>
            </div>
            """
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SMTP_EMAIL, SMTP_PASSWORD)
                server.sendmail(SMTP_EMAIL, user.email, msg.as_string())
        except Exception as e:
            print(f"Verification email failed: {e} — code for {user.email}: {code}")
    else:
        print(f"[VERIFY] {user.email} → {code}")

    return {"message": "Account created! Check your email for a verification code.", "email": user.email}

@app.post("/api/auth/verify-email/")
def verify_email(data: VerifyEmailSchema, db: Session = Depends(get_db)):
    key = f"verify_{data.email}"
    record = otp_store.get(key)
    if not record:
        raise HTTPException(status_code=400, detail="No verification code found. Please register again.")
    if dt.datetime.utcnow() > record["expires_at"]:
        otp_store.pop(key, None)
        raise HTTPException(status_code=400, detail="Code expired. Please register again.")
    if record["otp"] != data.code:
        raise HTTPException(status_code=400, detail="Invalid code.")

    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.is_verified = True
    db.commit()
    otp_store.pop(key, None)
    return {"message": "Email verified! You can now sign in."}


@app.post("/api/auth/login/")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.username == data.identifier) | (User.email == data.identifier)
    ).first()
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="EMAIL_NOT_VERIFIED")
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

    active = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()

    if active:
        # END the current session — mark it closed, keep its data
        active.is_active = False
        db.commit()
        return {"active": False, "session_code": None}
    else:
        # START — always create a brand-new row so each session has its own ID
        code     = _make_code()
        new_sess = ClassSession(
            teacher_id   = tp.id,
            session_code = code,
            is_active    = True,
            started_at   = dt.datetime.utcnow().isoformat(),
        )
        db.add(new_sess)
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
    """Full chronological log for a session: teacher CC + student AAC taps + student typed replies."""
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
    text_replies = []
    if student_ids:
        aac_logs = (
            db.query(AACLog)
            .filter(AACLog.session_id == session_id, AACLog.user_id.in_(student_ids))
            .order_by(AACLog.tapped_at.asc())
            .all()
        )
        text_replies = []  # Messages removed — communication via AAC Board → Live CC

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
    for r in text_replies:
        ts = r.sent_at or ""
        entries.append({
            "type":     "reply",
            "id":       f"reply-{r.id}",
            "sort_key": ts,
            "time":     ts[11:16] if len(ts) > 15 else ts,
            "speaker":  name_map.get(r.sender_id, f"Student #{r.sender_id}"),
            "text":     r.text,
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
        sess = db.query(ClassSession).filter_by(teacher_id=tid, is_active=True).first()
        if not sess:
            return []
        # Scope to current session only so old sessions don't bleed in
        msgs = (
            db.query(CCMessage)
            .filter(
                CCMessage.teacher_id == tid,
                CCMessage.id > since,
                CCMessage.sent_at >= sess.started_at,
            )
            .order_by(CCMessage.id.asc())
            .limit(30)
            .all()
        )
    else:
        # Teacher live-room: scope to current active session only
        tp = current_user.teacher_profile
        sess = db.query(ClassSession).filter_by(teacher_id=tp.id, is_active=True).first()
        if not sess:
            return []
        msgs = (
            db.query(CCMessage)
            .filter(
                CCMessage.teacher_id == tp.id,
                CCMessage.id > since,
                CCMessage.sent_at >= sess.started_at,
            )
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

# Messages removed — communication is now via AAC Board → Session Logs → Live CC


# ─────────────────────────────────────────────
# 11. ROSTER MANAGEMENT
# ─────────────────────────────────────────────

@app.post("/api/presence/")
def update_presence(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if profile:
        profile.last_seen = dt.datetime.utcnow().isoformat()
        db.commit()
    return {"message": "Presence updated"}

@app.get("/api/teacher/students/")
def get_my_students(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    profile = current_user.teacher_profile
    if not profile:
        raise HTTPException(status_code=404, detail="Teacher profile not found")

    now = dt.datetime.utcnow()
    result = []
    for s in profile.students:
        is_online = False
        if s.last_seen:
            try:
                last = dt.datetime.fromisoformat(s.last_seen)
                is_online = (now - last).total_seconds() < 120
            except Exception:
                pass
        result.append({
            "id": s.user_id,
            "username": s.user.username,
            "first_name": s.first_name or "",
            "last_name": s.last_name or "",
            "is_online": is_online,
            "status": "online" if is_online else "idle",
        })
    return result


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
def get_session_logs(
    since: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """AAC icon taps for the current active session; supports ?since=<last_id> for polling."""
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
            AACLog.id > since,
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