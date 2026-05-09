from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, DateTime, text, event
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from fastapi import Request
import requests
from passlib.context import CryptContext
import jwt
import datetime
import datetime as dt
import os
import io
import json
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. SETUP & CONFIG ---
SECRET_KEY = "your-super-secret-jwt-key"
ALGORITHM = "HS256"

HF_API_URL = "https://api-inference.huggingface.co/models/rammealz123/VOCALink-Mobile-STT"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

SMTP_EMAIL    = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
otp_store: dict = {}

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
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    status = Column(String, default="STUDENT")
    is_online = Column(Boolean, default=False)
    
    teacher_profile = relationship("TeacherProfile", back_populates="user", uselist=False)
    student_profile = relationship("StudentProfile", back_populates="user", uselist=False)

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

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    text = Column(String)
    is_aac = Column(Boolean, default=False)
    sent_at = Column(DateTime, default=dt.datetime.utcnow)

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

Base.metadata.create_all(bind=engine)

# --- AUTO-MIGRATION ---
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN is_online BOOLEAN DEFAULT FALSE"))
        conn.commit()
except Exception:
    pass

def create_user_profile_listener(mapper, connection, target):
    if target.status == "TEACHER":
        connection.execute(TeacherProfile.__table__.insert().values(user_id=target.id))
    elif target.status == "STUDENT":
        connection.execute(StudentProfile.__table__.insert().values(user_id=target.id))

event.listen(User, 'after_insert', create_user_profile_listener)

# --- 3. SCHEMAS ---
class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    status: str = "TEACHER"

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

class ChatMessageCreate(BaseModel):
    receiver_id: int
    text: str

class TTSSchema(BaseModel):
    text: str

class BroadcastSchema(BaseModel):
    text: str
    speaker: str = "teacher"

class ForgotPasswordSchema(BaseModel):
    email: str

class VerifyOTPSchema(BaseModel):
    email: str
    otp: str

class ResetPasswordSchema(BaseModel):
    email: str
    otp: str
    new_password: str

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

# --- 4. WEBSOCKET CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}

    async def connect(self, user_id: int, ws: WebSocket, db: Session):
        self.active_connections[user_id] = ws
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.is_online = True
            db.commit()
            await self.broadcast({"type": "STATUS_CHANGE", "user_id": user_id, "status": "online"})

    async def disconnect(self, user_id: int, db: Session):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.is_online = False
            db.commit()
            await self.broadcast({"type": "STATUS_CHANGE", "user_id": user_id, "status": "offline"})

    async def broadcast(self, message: dict):
        for user_id in list(self.active_connections.keys()):
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except:
                pass

    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

# --- 5. DEPENDENCIES & HELPERS ---
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

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
        if user is None: raise HTTPException(status_code=401)
        return user
    except:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

# --- 6. ROUTES ---

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    user_id = None
    try:
        token = await websocket.receive_text()
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        await manager.connect(user_id, websocket, db)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if user_id: await manager.disconnect(user_id, db)
    except:
        if user_id: await manager.disconnect(user_id, db)

# --- AUTH ROUTES ---
@app.post("/api/auth/register/")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == data.username) | (User.email == data.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already taken")
    new_user = User(username=data.username, email=data.email, hashed_password=pwd_context.hash(data.password), status=data.status)
    db.add(new_user); db.commit(); db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/api/auth/login/")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter((User.username == data.identifier) | (User.email == data.identifier)).first()
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user.is_online = True; db.commit()
    access_token = create_access_token(data={"user_id": user.id})
    return {"access_token": access_token, "status": user.status}

# --- FORGOT PASSWORD ---
@app.post("/api/auth/forgot-password/")
def forgot_password(data: ForgotPasswordSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user: return {"message": "If that email exists, an OTP has been sent."}
    otp = str(random.randint(100000, 999999))
    expires = dt.datetime.utcnow() + dt.timedelta(minutes=10)
    otp_store[data.email] = {"otp": otp, "expires_at": expires}
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "VocaLink OTP"; msg["From"] = SMTP_EMAIL; msg["To"] = data.email
        html = f"<div><h2>OTP: {otp}</h2></div>"
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, data.email, msg.as_string())
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    return {"message": "OTP sent"}

@app.post("/api/auth/verify-otp/")
def verify_otp(data: VerifyOTPSchema):
    record = otp_store.get(data.email)
    if not record or record["otp"] != data.otp or dt.datetime.utcnow() > record["expires_at"]:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")
    return {"message": "OTP verified."}

@app.post("/api/auth/reset-password/")
def reset_password(data: ResetPasswordSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user: raise HTTPException(status_code=404)
    user.hashed_password = pwd_context.hash(data.new_password); db.commit()
    return {"message": "Password reset successfully."}

# --- STUDENT MANAGEMENT ---
@app.get("/api/users/all-students/")
def get_all_students(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.status != "TEACHER": raise HTTPException(status_code=403)
    students = db.query(StudentProfile).join(User).filter(User.status == "STUDENT").all()
    return [{"id": s.user_id, "username": s.user.username, "first_name": s.first_name or "", "last_name": s.last_name or "", "assigned": s.instructor_id is not None, "status": "online" if s.user.is_online else "offline"} for s in students]

@app.get("/api/teacher/students/")
def get_my_students(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.status != "TEACHER": raise HTTPException(status_code=403)
    profile = current_user.teacher_profile
    if not profile: return []
    return [{"id": s.user_id, "username": s.user.username, "first_name": s.first_name or "", "last_name": s.last_name or "", "status": "online" if s.user.is_online else "offline"} for s in profile.students]

@app.post("/api/teacher/students/{user_id}")
def add_student_to_class(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.status != "TEACHER": raise HTTPException(status_code=403)
    student = db.query(StudentProfile).filter(StudentProfile.user_id == user_id).first()
    if not student: raise HTTPException(status_code=404)
    student.instructor_id = current_user.teacher_profile.id; db.commit()
    return {"message": "Student added"}

@app.delete("/api/teacher/students/{user_id}")
def remove_student_from_class(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    student = db.query(StudentProfile).filter(StudentProfile.user_id == user_id).first()
    if not student: raise HTTPException(status_code=404)
    student.instructor_id = None; db.commit()
    return {"message": "Student removed"}

# --- MESSAGES ---
@app.get("/api/messages/my-teacher")
def get_messages_from_teacher(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Fetches the chat history between the mobile student and their instructor."""
    if current_user.status != "STUDENT": raise HTTPException(status_code=403)
    p = current_user.student_profile
    if not p or not p.instructor_id: return []
    t_id = p.instructor.user_id
    return db.query(ChatMessage).filter(((ChatMessage.sender_id == current_user.id) & (ChatMessage.receiver_id == t_id)) | ((ChatMessage.sender_id == t_id) & (ChatMessage.receiver_id == current_user.id))).order_by(ChatMessage.sent_at.asc()).all()

@app.get("/api/messages/{target_id}")
def get_chat_history(target_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    messages = db.query(ChatMessage).filter(((ChatMessage.sender_id == current_user.id) & (ChatMessage.receiver_id == target_id)) | ((ChatMessage.sender_id == target_id) & (ChatMessage.receiver_id == current_user.id))).order_by(ChatMessage.sent_at.asc()).all()
    return messages

@app.post("/api/messages/")
async def send_chat_message(data: ChatMessageCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_msg = ChatMessage(sender_id=current_user.id, receiver_id=data.receiver_id, text=data.text)
    db.add(new_msg); db.commit()
    await manager.send_personal_message({"type": "NEW_MESSAGE", "sender_id": current_user.id, "text": data.text, "time": dt.datetime.utcnow().strftime("%H:%M")}, data.receiver_id)
    return {"message": "Sent"}

@app.post("/api/logs/")
async def log_and_message_aac(data: AACLogSchema, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    log = AACLog(user_id=current_user.id, icon_id=data.icon_id, icon_label=data.icon_label, message=data.message)
    db.add(log); student_profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if student_profile and student_profile.instructor_id:
        teacher_user_id = student_profile.instructor.user_id; text_content = data.message or f"Sent: {data.icon_label}"
        new_msg = ChatMessage(sender_id=current_user.id, receiver_id=teacher_user_id, text=text_content, is_aac=True); db.add(new_msg)
        await manager.send_personal_message({"type": "NEW_MESSAGE", "sender_id": current_user.id, "text": text_content, "is_aac": True, "time": dt.datetime.utcnow().strftime("%H:%M")}, teacher_user_id)
    db.commit(); return {"message": "Success"}

# --- TEACHER PROFILE ---
@app.get("/api/users/me/")
def get_me(user: User = Depends(get_current_user)):
    p = user.teacher_profile
    return {"username": user.username, "email": user.email, "first_name": p.first_name if p else "", "last_name": p.last_name if p else "", "display_name": p.display_name if p else "", "contact_number": p.contact_number if p else "", "room_section": p.room_section if p else "", "department": p.department if p else "", "grade_handled": p.grade_handled if p else "", "organization": p.organization if p else "", "bio": p.bio if p else ""}

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
    db.commit(); return {"message": "Profile updated"}

# --- STUDENT PROFILE ---
@app.get("/api/profile/me")
def get_profile(current_user: User = Depends(get_current_user)):
    if current_user.status == "TEACHER":
        p = current_user.teacher_profile
        return {"id": current_user.id, "username": current_user.username, "email": current_user.email, "status": current_user.status, "first_name": p.first_name if p else "", "last_name": p.last_name if p else "", "display_name": p.display_name if p else "", "department": p.department if p else "", "room_section": p.room_section if p else "", "bio": p.bio if p else ""}
    else:
        p = current_user.student_profile; teacher_name = ""; teacher_id = None
        if p and p.instructor:
            t = p.instructor
            teacher_name = f"{t.first_name} {t.last_name}".strip() or t.display_name or "Unknown Teacher"
            teacher_id = t.user_id
        return {"id": current_user.id, "username": current_user.username, "email": current_user.email, "status": current_user.status, "first_name": p.first_name if p else "", "last_name": p.last_name if p else "", "grade_level": p.grade_level if p else "", "disability_type": p.disability_type if p else "", "bio": p.bio if p else "", "teacher_name": teacher_name, "teacher_id": teacher_id}

@app.put("/api/profile/me")
def update_profile(profile_data: ProfileUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if not profile: raise HTTPException(status_code=404)
    if profile_data.first_name is not None: profile.first_name = profile_data.first_name
    if profile_data.last_name is not None: profile.last_name = profile_data.last_name
    if profile_data.bio is not None: profile.bio = profile_data.bio
    if profile_data.grade_level is not None: profile.grade_level = profile_data.grade_level
    if profile_data.disability_type is not None: profile.disability_type = profile_data.disability_type
    db.commit(); return {"message": "Success"}

# --- BROADCAST & CC ---
@app.post("/api/broadcast/")
def broadcast_message(data: BroadcastSchema, db: Session = Depends(get_db)):
    now = dt.datetime.utcnow().strftime("%H:%M")
    msg = CCMessage(text=data.text, speaker=data.speaker, sent_at=now); db.add(msg); db.commit(); return {"message": "Broadcasted"}

@app.get("/api/cc/messages/")
def get_cc_messages(since: int = 0, db: Session = Depends(get_db)):
    return db.query(CCMessage).filter(CCMessage.id > since).order_by(CCMessage.id.asc()).limit(20).all()

# --- TTS & STT ---
@app.post("/api/tts/")
def text_to_speech(data: TTSSchema):
    from gtts import gTTS
    tts = gTTS(text=data.text, lang='en'); buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0); return StreamingResponse(buf, media_type="audio/mpeg")

@app.post("/api/stt/")
async def speech_to_text(audio: UploadFile = File(...)):
    audio_bytes = await audio.read(); response = requests.post(HF_API_URL, headers=HF_HEADERS, data=audio_bytes); return response.json()