# Authentication Troubleshooting

## Current Authentication Methods

The app now supports **two authentication methods**:

1. **Supabase Auth** (primary) - Email-based authentication
2. **Custom Users Table** (fallback) - Username/email with password

## How to Test

### Option 1: Register a New User
1. Go to the registration page
2. Fill in:
   - Username
   - Email
   - Password
3. Click "Create Account"
4. The system will:
   - Try to create user in Supabase Auth
   - Also save to custom users table

### Option 2: Login with Existing User

**For Supabase Auth:**
- Use your **email** as the username
- Use your password

**For Custom Users Table:**
- Use your **username** OR **email**
- Use your password (may be plain text or hashed)

## Common Issues

1. **"Invalid credentials"** - Check:
   - Are you using the correct email/username?
   - Is the password correct?
   - Did you register first?

2. **No users in database** - You need to register first!

3. **Password mismatch** - If using custom users table:
   - Passwords may be stored as plain text (check your database)
   - Or as SHA256 hash

## Check Database

You can check if users exist in your database:
```sql
SELECT * FROM users;
```

Or check Supabase Auth users in the Supabase dashboard.











