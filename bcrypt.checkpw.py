# JHammit YAML Password hashing

# import library
import bcrypt

# Upload user passwords for main (jham) and secondary (test)
password1 = b'123'
password2 = b'456'

# Hashing passwords (1 and 2) using bcrypt
hashed_password1 = bcrypt.hashpw(password1, bcrypt.gensalt())
hashed_password2 = bcrypt.hashpw(password2, bcrypt.gensalt())

# Print hashed passwords to add to yaml file
print("Hashed Password 1:", hashed_password1)
print("Hashed Password 2:", hashed_password2)