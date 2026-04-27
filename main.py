from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, text
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from fastapi import Request
from passlib.context import CryptContext
import jwt
import datetime

# --- 1. SETUP & CONFIG ---
SECRET_KEY = "your-super-secret-jwt-key"
ALGORITHM = "HS256"

import os # Make sure this is imported at the top of your file!

# 1. Grab the Render Postgres URL (or fall back to local SQLite if on your laptop)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vocalink.db")

# 💥 Auto-migration hack

# 2. SQLAlchemy requires 'postgresql://' but Render gives 'postgres://', so we fix it:
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 3. Only use the SQLite specific arguments if we are actually using SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI(title="VocaLink API")

# Fix CORS so React can talk to it!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 2. DATABASE MODELS (SQLAlchemy) ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    status = Column(String, default="STUDENT")
    
    teacher_profile = relationship("TeacherProfile", back_populates="user", uselist=False)

class TeacherProfile(Base):
    __tablename__ = "teacher_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    first_name = Column(String, default="") # 💥 Moved here!
    last_name = Column(String, default="")  # 💥 Moved here!
    display_name = Column(String, default="")
    contact_number = Column(String, default="")
    room_section = Column(String, default="")
    department = Column(String, default="")

    user = relationship("User", back_populates="teacher_profile")   
    
Base.metadata.create_all(bind=engine)

try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE teacher_profiles ADD COLUMN first_name VARCHAR DEFAULT ''"))
        conn.execute(text("ALTER TABLE teacher_profiles ADD COLUMN last_name VARCHAR DEFAULT ''"))
        conn.commit()
except Exception as e:
    print(f"Migration check: {e}") # This will print the error instead of hiding it!

# --- 3. SCHEMAS (Pydantic) ---
class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    status: str = "TEACHER"

class LoginSchema(BaseModel):
    identifier: str # React sends this! Can be email or username
    password: str

class ProfileUpdateSchema(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    first_name: str | None = None  # 💥 ADD THIS!
    last_name: str | None = None
    display_name: str | None = None
    contact_number: str | None = None
    room_section: str | None = None
    department: str | None = None

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
    # Check if username or email exists
    if db.query(User).filter((User.username == data.username) | (User.email == data.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already taken")
    
    # Create User
    new_user = User(
        username=data.username,
        email=data.email,
        hashed_password=pwd_context.hash(data.password),
        status=data.status
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Automatically create the Teacher Profile (Replaces Django Signals!)
    if new_user.status == "TEACHER":
        profile = TeacherProfile(user_id=new_user.id)
        db.add(profile)
        db.commit()

    return {"message": "User created successfully"}

@app.post("/api/auth/login/")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    # Find by username OR email
    user = db.query(User).filter((User.username == data.identifier) | (User.email == data.identifier)).first()
    
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    access_token = create_access_token(data={"user_id": user.id})
    return {"access": access_token, "status": user.status}

@app.get("/api/users/me/")
def get_me(user: User = Depends(get_current_user)):
    profile = user.teacher_profile
    return {
        "username": user.username,
        "email": user.email,
        "first_name": profile.first_name if profile else "", # 💥 Point to profile
        "last_name": profile.last_name if profile else "",   # 💥 Point to profile
        "display_name": profile.display_name if profile else "",
        "contact_number": profile.contact_number if profile else "",
        "room_section": profile.room_section if profile else "",
        "department": profile.department if profile else "",
    }

@app.patch("/api/users/me/")
def update_me(data: ProfileUpdateSchema, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Update core user info
    if data.username: user.username = data.username
    if data.email: user.email = data.email
    
    # Update profile info
    if user.teacher_profile:
        if data.first_name is not None: user.teacher_profile.first_name = data.first_name # 💥 Point to profile
        if data.last_name is not None: user.teacher_profile.last_name = data.last_name   # 💥 Point to profile
        if data.display_name is not None: user.teacher_profile.display_name = data.display_name
        if data.contact_number is not None: user.teacher_profile.contact_number = data.contact_number
        if data.room_section is not None: user.teacher_profile.room_section = data.room_section
        if data.department is not None: user.teacher_profile.department = data.department

    db.commit()
    return {"message": "Profile updated"}