# MediNLP Update Notes

Added in this version:

1. Red-flag symptom display
   - Backend now returns `triggered_symptoms` inside `red_flag_result`.
   - Frontend shows the exact red-flag symptom chips that triggered the alert.

2. Login, guest mode, and chat restore
   - SQLite tables added according to the ER idea:
     - `users(id, first_name, last_name, name, email, password)`
     - `chat(id, user_id, title, messages_json)`
     - `report(id, user_id, chat_id, result)`
   - Guest users can chat, but history is not restored after refresh.
   - Logged-in users get saved chat history in the sidebar.

3. Voice mode update
   - GPT-style mic SVG button.
   - English/Banglish and Bangla voice language selector.
   - More alternatives checked from browser speech recognition.
   - Composer layout fixed so the mic/status area does not shift when clicked.

Run commands:

Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```
