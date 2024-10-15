# New user registration could be activated if desired
        if st.button("Register New User"):
            try:
                reg_result = authenticator.register_user()
                if len(reg_result) == 3:
                    email_of_registered_user, username_of_registered_user, name_of_registered_user = reg_result
                    st.success('User registered successfully')
                else:
                    st.error('Registration failed. Please try again.')
            except RegisterError as e:
                st.error(e)

        # Forgot password could be activated if desired
        if st.button("Forgot Password"):
            try:
                forgot_result = authenticator.forgot_password()
                if len(forgot_result) == 3:
                    username_of_forgotten_password, email_of_forgotten_password, new_random_password = forgot_result
                    st.success('New password sent securely')
                else:
                    st.error('Error: Username not found')
            except ForgotError as e:
                st.error(e)

        # Forgot username could be activated if desired
        if st.button("Forgot Username"):
            try:
                username_of_forgotten_username, email_of_forgotten_username = authenticator.forgot_username()
                if username_of_forgotten_username:
                    st.success('Username sent securely')
                else:
                    st.error('Email not found')
            except ForgotError as e:
                st.error(e)