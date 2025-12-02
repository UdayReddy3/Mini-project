"""
Setup script to create demo user for testing
"""

from db import register_user, init_database

# Initialize database
init_database()

# Create demo user
success, message = register_user(
    username="demo",
    email="demo@plantdisease.com",
    password="demo123",
    full_name="Demo User"
)

print(message)

if success:
    print("\n✓ Demo user created!")
    print("Login credentials:")
    print("  Username: demo")
    print("  Password: demo123")
else:
    print("\nDemo user already exists or error occurred")
