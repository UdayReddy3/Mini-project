"""
Database Module for User Authentication
Handles user registration, login, and password management using SQLite.
"""

import sqlite3
import hashlib
import os
from datetime import datetime


DATABASE_PATH = 'users.db'


def init_database():
    """Initialize the SQLite database with users table."""
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")


def hash_password(password):
    """
    Hash a password using SHA256.
    
    Args:
        password (str): Plain text password
        
    Returns:
        str: Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, email, password, full_name=""):
    """
    Register a new user in the database.
    
    Args:
        username (str): Username
        email (str): Email address
        password (str): Plain text password
        full_name (str): Full name (optional)
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Validate inputs
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if not email or '@' not in email:
            return False, "Invalid email format"
        
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        # Check if user already exists
        c.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if c.fetchone():
            return False, "Username or email already exists"
        
        # Hash password and insert user
        password_hash = hash_password(password)
        c.execute('''
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, full_name))
        
        conn.commit()
        conn.close()
        
        return True, f"✓ User '{username}' registered successfully!"
    
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, f"Registration error: {str(e)}"


def login_user(username, password):
    """
    Authenticate a user and update last login.
    
    Args:
        username (str): Username
        password (str): Plain text password
        
    Returns:
        tuple: (success: bool, message: str, user_id: int or None)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Get user from database
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        
        if not user:
            return False, "Username not found", None
        
        user_id, stored_hash = user
        password_hash = hash_password(password)
        
        if password_hash != stored_hash:
            return False, "Incorrect password", None
        
        # Update last login
        c.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now(), user_id))
        conn.commit()
        conn.close()
        
        return True, f"✓ Welcome back, {username}!", user_id
    
    except Exception as e:
        return False, f"Login error: {str(e)}", None


def get_user_info(user_id):
    """
    Get user information by user ID.
    
    Args:
        user_id (int): User ID
        
    Returns:
        dict: User information or None
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        c.execute('SELECT id, username, email, full_name, created_at FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'created_at': user[4]
            }
        return None
    
    except Exception as e:
        print(f"Error fetching user info: {str(e)}")
        return None


def change_password(user_id, old_password, new_password):
    """
    Change user password.
    
    Args:
        user_id (int): User ID
        old_password (str): Current password
        new_password (str): New password
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Verify old password
        c.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
        result = c.fetchone()
        
        if not result:
            return False, "User not found"
        
        old_hash = result[0]
        if hash_password(old_password) != old_hash:
            return False, "Current password is incorrect"
        
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters"
        
        # Update password
        new_hash = hash_password(new_password)
        c.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
        conn.commit()
        conn.close()
        
        return True, "✓ Password changed successfully!"
    
    except Exception as e:
        return False, f"Error: {str(e)}"


def delete_user(user_id, password):
    """
    Delete a user account.
    
    Args:
        user_id (int): User ID
        password (str): Password for confirmation
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Verify password
        c.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
        result = c.fetchone()
        
        if not result:
            return False, "User not found"
        
        if hash_password(password) != result[0]:
            return False, "Password is incorrect"
        
        # Delete user
        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        return True, "✓ Account deleted successfully!"
    
    except Exception as e:
        return False, f"Error: {str(e)}"


# Initialize database on module import
if not os.path.exists(DATABASE_PATH):
    init_database()
