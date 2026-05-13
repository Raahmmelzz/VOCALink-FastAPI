from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, text, event, Boolean
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from fastapi import Request

import requests
from passlib.context import CryptContext
import jwt
import datetime
import os
import io
import tempfile
import json
import random
import string

# --- 1. SETUP & CONFIG ---
SECRET_KEY = "your-super-secret-jwt-key"
ALGORITHM = "HS256"
active_sessions = {}
HF_API_URL = "https://api-inference.huggingface.co/models/rammealz123/VOCALink-Mobile-STT"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vocalink.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI(title="VocaLink API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DATABASE MODELS ---
import datetime as dt

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    status = Column(String, default="STUDENT")

    teacher_profile = relationship("TeacherProfile", back_populates="user", uselist=False)
    student_profile = relationship("StudentProfile", back_populates="user", uselist=False)
    sent_messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.receiver_id", back_populates="receiver")

class TeacherProfile(Base):
    __tablename__ = "teacher_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    first_name = Column(String, default="")
    last_name = Column(String, default="")
    display_name = Column(String, default="")
    contact_number = Column(String, default="")
    room_section = Column(String, default="")
    department = Column(String, default="")
    grade_handled = Column(String, default="")
    organization = Column(String, default="")
    bio = Column(String, default="")

    user = relationship("User", back_populates="teacher_profile")
    students = relationship("StudentProfile", back_populates="instructor")

class StudentProfile(Base):
    __tablename__ = "student_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    instructor_id = Column(Integer, ForeignKey("teacher_profiles.id", ondelete="SET NULL"), nullable=True)

    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    grade_level = Column(String, nullable=True)
    disability_type = Column(String, nullable=True)

    instructor = relationship("TeacherProfile", back_populates="students")
    user = relationship("User", back_populates="student_profile")

class AACLog(Base):
    __tablename__ = "aac_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    icon_id = Column(String)
    icon_label = Column(String)
    message = Column(String, nullable=True)
    tapped_at = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

class CCMessage(Base):
    __tablename__ = "cc_messages"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    speaker = Column(String, default="teacher")
    sent_at = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

# ✅ NEW: Direct message model for the Messages screen
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    receiver_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    text = Column(String)
    is_aac = Column(Boolean, default=False)
    sent_at = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="received_messages")

Base.metadata.create_all(bind=engine)

# Auto-migration for existing databases
columns_to_add_teacher = [
    "first_name VARCHAR DEFAULT ''", "last_name VARCHAR DEFAULT ''",
    "grade_handled VARCHAR DEFAULT ''", "organization VARCHAR DEFAULT ''", "bio VARCHAR DEFAULT ''"
]
for column in columns_to_add_teacher:
    try:
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE teacher_profiles ADD COLUMN {column}"))
            conn.commit()
    except Exception:
        pass

# Auto-migrate messages table columns if they don't exist
messages_columns = [
    "sender_id INTEGER", "receiver_id INTEGER", "text VARCHAR",
    "is_aac BOOLEAN DEFAULT 0", "sent_at VARCHAR"
]
for column in messages_columns:
    try:
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE messages ADD COLUMN {column}"))
            conn.commit()
    except Exception:
        pass

# Profile creation handled in register endpoint — no event listener needed

# --- 3. SCHEMAS (Pydantic) ---
class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    status: str = "STUDENT"

class ProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    bio: Optional[str] = None
    grade_level: Optional[str] = None
    disability_type: Optional[str] = None

class LoginSchema(BaseModel):
    identifier: str
    password: str

class AACLogSchema(BaseModel):
    icon_id: str
    icon_label: str
    message: Optional[str] = None

class TTSSchema(BaseModel):
    text: str

class BroadcastSchema(BaseModel):
    text: str
    speaker: str = "teacher"

# ✅ NEW: Schema for sending a direct message
class MessageSchema(BaseModel):
    receiver_id: int
    text: str
    is_aac: bool = False

class ProfileUpdateSchema(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    contact_number: str | None = None
    room_section: str | None = None
    department: str | None = None
    grade_handled: str | None = None
    organization: str | None = None
    bio: str | None = None

# --- WEBSOCKET CONNECTION MANAGER ---
# ✅ FIX: Only ONE manager now. The old `manager` (ConnectionManager) is removed.
# Both teacher and students connect through this single LiveRoomManager.
class LiveRoomManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}

    async def connect(self, ws: WebSocket, user_id: int):
        self.active_connections[user_id] = ws
        await self.broadcast_presence()

    async def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            await self.broadcast_presence()

    async def broadcast_presence(self):
        online_count = len(self.active_connections)
        await self.broadcast({"type": "presence", "count": online_count})

    async def broadcast(self, message: dict):
        dead_connections = []
        for uid, ws in self.active_connections.items():
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead_connections.append(uid)
        for uid in dead_connections:
            if uid in self.active_connections:
                del self.active_connections[uid]

room_manager = LiveRoomManager()

# --- 4. DEPENDENCIES & HELPERS ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=1)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token.split(" ")[1], SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(User).filter(User.id == payload.get("user_id")).first()
        if user is None:
            raise HTTPException(status_code=401)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- 5. WEBSOCKET ENDPOINT ---
# ✅ FIX: Only ONE /ws/cc route. Both teacher and students land in the same room_manager.
@app.websocket("/ws/cc")
async def websocket_cc(websocket: WebSocket):
    await websocket.accept()
    user_id = None
    try:
        token = await websocket.receive_text()
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")

        if not user_id:
            await websocket.close(code=1008)
            return

        await room_manager.connect(websocket, user_id)

        while True:
            await websocket.receive_text()  # keep connection alive

    except WebSocketDisconnect:
        if user_id:
            await room_manager.disconnect(user_id)
    except Exception:
        if user_id:
            await room_manager.disconnect(user_id)
        try:
            await websocket.close()
        except:
            pass

# --- 6. BROADCAST ---
# ✅ FIX: Now requires auth and only teachers can broadcast.
@app.post("/api/broadcast/")
async def broadcast_to_students(
    data: BroadcastSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Only teachers can broadcast.")

    now = datetime.datetime.utcnow().strftime("%H:%M")

    msg = CCMessage(text=data.text, speaker=data.speaker, sent_at=now)
    db.add(msg)
    db.commit()

    await room_manager.broadcast({
        "type": "message",
        "text": data.text,
        "speaker": data.speaker,
        "time": now
    })
    return {"message": "Broadcasted"}

# --- 7. SESSION ROUTES ---
@app.post("/api/sessions/toggle")
def toggle_session(current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Only teachers can manage sessions.")

    teacher_id = current_user.id
    if teacher_id in active_sessions:
        del active_sessions[teacher_id]
        return {"active": False, "session_code": None}
    else:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        active_sessions[teacher_id] = code
        return {"active": True, "session_code": code}

# ✅ NEW: Teachers can check if their own session is still running after navigating away
@app.get("/api/sessions/teacher")
def check_teacher_session(current_user: User = Depends(get_current_user)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only.")
    if current_user.id in active_sessions:
        return {"active": True, "session_code": active_sessions[current_user.id]}
    return {"active": False, "session_code": None}

@app.get("/api/sessions/student")
def check_student_session(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.status != "STUDENT":
        return {"active": False}
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if not profile or not profile.instructor_id:
        return {"active": False}

    instructor_profile = db.query(TeacherProfile).filter(TeacherProfile.id == profile.instructor_id).first()
    if not instructor_profile:
        return {"active": False}

    if instructor_profile.user_id in active_sessions:
        return {"active": True, "session_code": active_sessions[instructor_profile.user_id]}
    return {"active": False}

@app.get("/api/cc/messages/")
def get_cc_messages(since: int = 0, db: Session = Depends(get_db)):
    return db.query(CCMessage).filter(CCMessage.id > since).order_by(CCMessage.id.asc()).limit(20).all()

# --- 8. AUTH ROUTES ---
@app.post("/api/auth/register/")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == data.username) | (User.email == data.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already taken")

    new_user = User(
        username=data.username,
        email=data.email,
        hashed_password=pwd_context.hash(data.password),
        status=data.status
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    if new_user.status == "TEACHER":
        profile = TeacherProfile(user_id=new_user.id)
        db.add(profile)
    elif new_user.status == "STUDENT":
        profile = StudentProfile(user_id=new_user.id)
        db.add(profile)

    db.commit()
    return {"message": "User created successfully"}

@app.post("/api/auth/login/")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter((User.username == data.identifier) | (User.email == data.identifier)).first()

    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"user_id": user.id})
    return {"access_token": access_token, "status": user.status}

# --- 9. PROFILE ROUTES ---
@app.get("/api/profile/me")
def get_profile(current_user: User = Depends(get_current_user)):
    if current_user.status == "TEACHER":
        p = current_user.teacher_profile
        return {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "status": current_user.status,
            "first_name": p.first_name if p else "",
            "last_name": p.last_name if p else "",
            "display_name": p.display_name if p else "",
            "department": p.department if p else "",
            "grade_handled": p.grade_handled if p else "",
            "room_section": p.room_section if p else "",
            "bio": p.bio if p else "",
        }
    else:
        p = current_user.student_profile
        # Include teacher info so Messages screen can find the teacher
        teacher_name = ""
        teacher_id = None
        if p and p.instructor_id:
            instructor = p.instructor
            if instructor:
                teacher_name = instructor.first_name or instructor.display_name or "Teacher"
                teacher_id = instructor.user_id
        return {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "status": current_user.status,
            "first_name": p.first_name if p else "",
            "last_name": p.last_name if p else "",
            "grade_level": p.grade_level if p else "",
            "disability_type": p.disability_type if p else "",
            "bio": p.bio if p else "",
            "teacher_name": teacher_name,
            "teacher_id": teacher_id,
        }

@app.put("/api/profile/me")
def update_profile(
    profile_data: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if profile_data.first_name is not None: profile.first_name = profile_data.first_name
    if profile_data.last_name is not None: profile.last_name = profile_data.last_name
    if profile_data.bio is not None: profile.bio = profile_data.bio
    if profile_data.grade_level is not None: profile.grade_level = profile_data.grade_level
    if profile_data.disability_type is not None: profile.disability_type = profile_data.disability_type

    db.commit()
    return {"message": "Profile updated successfully!"}

@app.delete("/api/profile/me")
def delete_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db.delete(current_user)
    db.commit()
    return {"message": "Account permanently deleted."}

@app.get("/api/users/me/")
def get_me(user: User = Depends(get_current_user)):
    profile = user.teacher_profile
    return {
        "username": user.username,
        "email": user.email,
        "first_name": profile.first_name if profile else "",
        "last_name": profile.last_name if profile else "",
        "display_name": profile.display_name if profile else "",
        "contact_number": profile.contact_number if profile else "",
        "room_section": profile.room_section if profile else "",
        "department": profile.department if profile else "",
        "grade_handled": profile.grade_handled if profile else "",
        "organization": profile.organization if profile else "",
        "bio": profile.bio if profile else "",
    }

@app.patch("/api/users/me/")
def update_me(data: ProfileUpdateSchema, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if data.username: user.username = data.username
    if data.email: user.email = data.email

    if user.teacher_profile:
        if data.first_name is not None: user.teacher_profile.first_name = data.first_name
        if data.last_name is not None: user.teacher_profile.last_name = data.last_name
        if data.display_name is not None: user.teacher_profile.display_name = data.display_name
        if data.contact_number is not None: user.teacher_profile.contact_number = data.contact_number
        if data.room_section is not None: user.teacher_profile.room_section = data.room_section
        if data.department is not None: user.teacher_profile.department = data.department
        if data.grade_handled is not None: user.teacher_profile.grade_handled = data.grade_handled
        if data.organization is not None: user.teacher_profile.organization = data.organization
        if data.bio is not None: user.teacher_profile.bio = data.bio

    db.commit()
    return {"message": "Profile updated"}

# --- 10. DIRECT MESSAGES ROUTES ---
# ✅ NEW: These power the Messages screen

@app.post("/api/messages/")
def send_message(
    data: MessageSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    msg = Message(
        sender_id=current_user.id,
        receiver_id=data.receiver_id,
        text=data.text,
        is_aac=data.is_aac,
        sent_at=dt.datetime.utcnow().isoformat(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"id": msg.id, "message": "Sent"}

@app.get("/api/messages/my-teacher")
def get_messages_with_teacher(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns full conversation between the student and their assigned teacher."""
    if current_user.status == "STUDENT":
        profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
        if not profile or not profile.instructor_id:
            return []
        teacher_user_id = profile.instructor.user_id
        other_id = teacher_user_id
    else:
        # Teacher calling this: not typical, return empty
        return []

    messages = db.query(Message).filter(
        ((Message.sender_id == current_user.id) & (Message.receiver_id == other_id)) |
        ((Message.sender_id == other_id) & (Message.receiver_id == current_user.id))
    ).order_by(Message.sent_at.asc()).all()

    return [
        {
            "id": m.id,
            "sender_id": m.sender_id,
            "receiver_id": m.receiver_id,
            "text": m.text,
            "is_aac": m.is_aac,
            "sent_at": m.sent_at,
        }
        for m in messages
    ]

@app.get("/api/messages/my-students")
def get_messages_from_students(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns all messages sent to this teacher by any of their students."""
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only.")

    profile = current_user.teacher_profile
    if not profile:
        return []

    student_user_ids = [s.user_id for s in profile.students]

    messages = db.query(Message).filter(
        ((Message.sender_id.in_(student_user_ids)) & (Message.receiver_id == current_user.id)) |
        ((Message.sender_id == current_user.id) & (Message.receiver_id.in_(student_user_ids)))
    ).order_by(Message.sent_at.asc()).all()

    return [
        {
            "id": m.id,
            "sender_id": m.sender_id,
            "receiver_id": m.receiver_id,
            "text": m.text,
            "is_aac": m.is_aac,
            "sent_at": m.sent_at,
        }
        for m in messages
    ]

# --- 11. STUDENT/TEACHER MANAGEMENT ---
@app.get("/api/teacher/students/")
def get_my_students(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    profile = current_user.teacher_profile
    if not profile:
        raise HTTPException(status_code=404, detail="Teacher profile not found")
    return [
        {
            "id": s.user_id,
            "username": s.user.username,
            "first_name": s.first_name or "",
            "last_name": s.last_name or "",
            "status": "offline"
        }
        for s in profile.students
    ]

@app.get("/api/users/all-students/")
def get_all_students(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    students = db.query(StudentProfile).join(User).all()
    return [
        {
            "id": s.user_id,
            "username": s.user.username,
            "first_name": s.first_name or "",
            "last_name": s.last_name or "",
            "assigned": s.instructor_id is not None,
            "status": "offline"
        }
        for s in students
    ]

@app.post("/api/teacher/students/{user_id}")
def add_student_to_class(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.status != "TEACHER":
        raise HTTPException(status_code=403, detail="Teachers only")
    student = db.query(StudentProfile).filter(StudentProfile.user_id == user_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student.instructor_id = current_user.teacher_profile.id
    db.commit()
    return {"message": "Student added to class"}

@app.delete("/api/teacher/students/{user_id}")
def remove_student_from_class(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    student = db.query(StudentProfile).filter(StudentProfile.user_id == user_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student.instructor_id = None
    db.commit()
    return {"message": "Student removed"}

# --- 12. LOGS ---
@app.post("/api/logs/")
def log_icon_tap(
    data: AACLogSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    log = AACLog(
        user_id=current_user.id,
        icon_id=data.icon_id,
        icon_label=data.icon_label,
        message=data.message,
        tapped_at=dt.datetime.utcnow().isoformat(),
    )
    db.add(log)
    db.commit()
    return {"message": "Log saved."}

@app.get("/api/logs/")
def get_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logs = db.query(AACLog).filter(AACLog.user_id == current_user.id)\
              .order_by(AACLog.id.desc()).limit(50).all()
    return [
        {
            "id": l.id,
            "icon_id": l.icon_id,
            "icon_label": l.icon_label,
            "message": l.message,
            "tapped_at": l.tapped_at,
        }
        for l in logs
    ]

# --- 13. TTS ---
@app.post("/api/tts/")
def text_to_speech(
    data: TTSSchema,
    current_user: User = Depends(get_current_user)
):
    try:
        from gtts import gTTS
        tts = gTTS(text=data.text, lang='en')
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

# --- 14. STT ---
@app.post("/api/stt/")
async def speech_to_text(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not audio.filename.lower().endswith('.wav') and audio.content_type != 'audio/wav':
        raise HTTPException(
            status_code=400,
            detail="Invalid file format! This AI only accepts .wav audio files."
        )

    try:
        audio_bytes = await audio.read()

        try:
            response = requests.post(HF_API_URL, headers=HF_HEADERS, data=audio_bytes, timeout=30)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Could not reach Hugging Face: {str(e)}")

        output = response.json()

        if response.status_code == 503 or (isinstance(output, dict) and "estimated_time" in output):
            return {
                "error": "Model is warming up",
                "estimated_time": output.get("estimated_time", 20),
                "message": "The AI is waking up. Please try again in 20 seconds!"
            }

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"HF Error: {output}")

        if isinstance(output, list) and len(output) > 0:
            return {"text": output[0].get("text", "No transcription available")}

        return {"text": output.get("text", str(output))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT Error: {str(e)}")