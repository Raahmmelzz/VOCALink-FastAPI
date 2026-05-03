from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect
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
import os
import io
import tempfile
import json 

# --- 1. SETUP & CONFIG ---
SECRET_KEY = "your-super-secret-jwt-key"
ALGORITHM = "HS256"

HF_API_URL = "https://api-inference.huggingface.co/models/rammealz123/VOCALink-Mobile-STT"
# 🚨 Replace this with your actual token:
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

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

# --- 2. DATABASE MODELS (SQLAlchemy) ---
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

# 💥 MOVED THIS UP: Now SQLAlchemy knows about it BEFORE it creates tables!
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

# 💥 Tell the database to build all the tables NOW.
Base.metadata.create_all(bind=engine)

# 💥 THE BULLETPROOF AUTO-MIGRATION HACK
# Keeping your awesome hack to add missing columns to existing databases!
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
    
def create_user_profile_listener(mapper, connection, target):
    # 'target' is the newly created User object
    if target.status == "TEACHER":
        # We use connection.execute to safely insert during the event
        connection.execute(
            TeacherProfile.__table__.insert().values(user_id=target.id)
        )
    elif target.status == "STUDENT":
        connection.execute(
            StudentProfile.__table__.insert().values(user_id=target.id)
        )

# Attach the listener to the User model
event.listen(User, 'after_insert', create_user_profile_listener)

# --- 3. SCHEMAS (Pydantic) ---
class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    status: str = "TEACHER" # Note: Mobile app currently overrides this to "STUDENT"
    
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

# --- WEBSOCKET CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

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

# --- 5. ROUTES ---
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

    # 💥 FIXED: Now it automatically builds the CORRECT profile based on status!
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
    # 💥 FIXED: Changed "access" to "access_token" to match the React Native code!
    return {"access_token": access_token, "status": user.status}

@app.post("/api/stt/")
async def speech_to_text(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    # 1. The WAV Bouncer (Keep this!)
    if not audio.filename.lower().endswith('.wav') and audio.content_type != 'audio/wav':
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format! This AI only accepts .wav audio files."
        )

    try:
        audio_bytes = await audio.read()
        
        # 2. Try to hit the Cloud AI
        try:
            response = requests.post(HF_API_URL, headers=HF_HEADERS, data=audio_bytes, timeout=30)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Could not reach Hugging Face: {str(e)}")

        output = response.json()

        # 3. Handle the "Model is Loading" state (Free Tier common issue)
        if response.status_code == 503 or (isinstance(output, dict) and "estimated_time" in output):
            return {
                "error": "Model is warming up",
                "estimated_time": output.get("estimated_time", 20),
                "message": "The AI is waking up. Please try again in 20 seconds!"
            }

        # 4. Handle actual errors
        if response.status_code != 200:
             raise HTTPException(status_code=response.status_code, detail=f"HF Error: {output}")

        # 5. Success!
        if isinstance(output, list) and len(output) > 0:
            return {"text": output[0].get("text", "No transcription available")}
        
        return {"text": output.get("text", str(output))}

    except Exception as e:
        # This catches the error you just saw and provides a real message instead
        raise HTTPException(status_code=500, detail=f"STT Error: {str(e)}")
    
# --- TEACHER ROUTES ---
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

# --- STUDENT ROUTES ---
# 💥 FIXED: Added the /api prefix so React Native can find it!
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

# 💥 FIXED: Added the /api prefix so React Native can find it!
@app.delete("/api/profile/me")
def delete_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db.delete(current_user)
    db.commit()
    return {"message": "Account permanently deleted."}

# --- SHARED PROFILE GET (works for both student and teacher) ---
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

# --- PHASE 4: WEBSOCKET (Live CC) ---
@app.websocket("/ws/cc")
async def websocket_cc(websocket: WebSocket):
    # Accept first, then validate token sent as first message
    await websocket.accept()
    try:
        # Wait for token as first message
        token = await websocket.receive_text()
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        db = SessionLocal()
        user = db.query(User).filter(User.id == payload.get("user_id")).first()
        db.close()
        if not user:
            await websocket.close(code=1008)
            return
    except Exception:
        await websocket.close(code=1008)
        return

    manager.active.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/broadcast/")
async def broadcast_to_students(
    data: BroadcastSchema,
    current_user: User = Depends(get_current_user)
):
    now = datetime.datetime.now().strftime("%H:%M")
    await manager.broadcast({
        "text": data.text,
        "speaker": data.speaker,
        "time": now,
    })
    return {"message": f"Broadcasted to {len(manager.active)} student(s)"}

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

# --- PHASE 3: STT (Whisper) ---
# Lazy-load the model so the server starts fast
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model


"""
@app.post("/api/stt/")
async def speech_to_text(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    try:
        # Save uploaded audio to a temp file
        suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        try:
            model = get_whisper_model()
            segments, _ = model.transcribe(tmp_path, language="en")
            text = " ".join(seg.text.strip() for seg in segments)
            return {"text": text.strip()}
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")
        
"""