from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, text
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
import tempfile
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

class AACLog(Base):
    __tablename__ = "aac_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    icon_id = Column(String)
    icon_label = Column(String)
    message = Column(String, nullable=True)
    tapped_at = Column(String, default=lambda: dt.datetime.utcnow().isoformat())

Base.metadata.create_all(bind=engine)

# Auto-migration for missing columns
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

# Profile creation is handled explicitly in the register endpoint

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

class TTSSchema(BaseModel):
    text: str

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

# --- 4. DEPENDENCIES & HELPERS ---
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
        if user is None:
            raise HTTPException(status_code=401)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- 5. AUTH ROUTES ---
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
        db.add(TeacherProfile(user_id=new_user.id))
    elif new_user.status == "STUDENT":
        db.add(StudentProfile(user_id=new_user.id))
    db.commit()
    return {"message": "User created successfully"}

@app.post("/api/auth/login/")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter((User.username == data.identifier) | (User.email == data.identifier)).first()
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"user_id": user.id})
    return {"access_token": access_token, "status": user.status}

# --- FORGOT PASSWORD ---
@app.post("/api/auth/forgot-password/")
def forgot_password(data: ForgotPasswordSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        # Don't reveal if email exists — always return success
        return {"message": "If that email exists, an OTP has been sent."}

    otp = str(random.randint(100000, 999999))
    expires = dt.datetime.utcnow() + dt.timedelta(minutes=10)
    otp_store[data.email] = {"otp": otp, "expires_at": expires}

    # Try to send email — if SMTP not configured, log OTP instead
    if SMTP_EMAIL and SMTP_PASSWORD:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "VocaLink Password Reset OTP"
            msg["From"]    = SMTP_EMAIL
            msg["To"]      = data.email
            html = f"""
            <div style="font-family:sans-serif;max-width:480px;margin:auto;padding:32px;background:#f9f9f9;border-radius:12px">
              <h2 style="color:#1AADDC">VocaLink — Password Reset</h2>
              <p>Your one-time password (OTP) is:</p>
              <div style="font-size:36px;font-weight:800;letter-spacing:8px;color:#1A1A2E;padding:16px 0">{otp}</div>
              <p style="color:#6B7280;font-size:13px">This OTP expires in <strong>10 minutes</strong>.</p>
            </div>
            """
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SMTP_EMAIL, SMTP_PASSWORD)
                server.sendmail(SMTP_EMAIL, data.email, msg.as_string())
        except Exception as e:
            print(f"Email failed: {e} — OTP for {data.email}: {otp}")
    else:
        # No SMTP configured — print OTP to Render logs for testing
        print(f"[OTP] {data.email} → {otp}")

    return {"message": "OTP sent to your email."}

@app.post("/api/auth/verify-otp/")
def verify_otp(data: VerifyOTPSchema):
    record = otp_store.get(data.email)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP found. Please request a new one.")
    if dt.datetime.utcnow() > record["expires_at"]:
        otp_store.pop(data.email, None)
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
    if record["otp"] != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP.")
    return {"message": "OTP verified."}

@app.post("/api/auth/reset-password/")
def reset_password(data: ResetPasswordSchema, db: Session = Depends(get_db)):
    record = otp_store.get(data.email)
    if not record or record["otp"] != data.otp or dt.datetime.utcnow() > record["expires_at"]:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.hashed_password = pwd_context.hash(data.new_password)
    db.commit()
    otp_store.pop(data.email, None)
    return {"message": "Password reset successfully."}

# --- TEACHER ROUTES ---
@app.get("/api/users/me/")
def get_me(user: User = Depends(get_current_user)):
    p = user.teacher_profile
    return {
        "username": user.username,
        "email": user.email,
        "first_name": p.first_name if p else "",
        "last_name": p.last_name if p else "",
        "display_name": p.display_name if p else "",
        "contact_number": p.contact_number if p else "",
        "room_section": p.room_section if p else "",
        "department": p.department if p else "",
        "grade_handled": p.grade_handled if p else "",
        "organization": p.organization if p else "",
        "bio": p.bio if p else "",
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

# --- STUDENT ROUTES ---
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
            "room_section": p.room_section if p else "",
            "bio": p.bio if p else "",
        }
    else:
        p = current_user.student_profile
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

# --- ICON TAP LOGS ---
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

# --- PHASE 3: TTS (gTTS) ---
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

# --- PHASE 3: STT (HuggingFace Whisper) ---
@app.post("/api/stt/")
async def speech_to_text(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not audio.filename.lower().endswith('.wav') and audio.content_type != 'audio/wav':
        raise HTTPException(status_code=400, detail="Only .wav files accepted.")
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
