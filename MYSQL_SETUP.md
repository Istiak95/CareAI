# MediNLP MySQL Chat Restore Setup

The backend now supports MySQL for login, saved chats, restored chat history, and reports.

## 1. Install MySQL

Install MySQL Server and make sure it is running.

## 2. Configure backend `.env`

Open `backend/.env` and add/update:

```env
DB_PROVIDER=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=medinlp_chatbot
AUTH_SECRET=change-this-secret-before-deployment
TOKEN_TTL_SECONDS=1209600
```

The FastAPI backend will create the `medinlp_chatbot` database automatically if the MySQL user has permission.

## 3. Install dependencies

```bash
cd backend
python -m pip install -r requirements.txt
```

## 4. Run backend

```bash
python -m uvicorn main:app --reload --port 8000
```

## 5. Tables created automatically

- `users`: registered users and password hashes
- `chat`: saved chat sessions and messages JSON
- `report`: saved prediction/report results

## Guest mode behavior

Guest mode does not save or restore chats. Login is required for chat restore.
