# How to Create a GitHub Personal Access Token with 2FA

Since GitHub requires tokens instead of passwords, here's how to create one using your authenticator:

## Step 1: Create Personal Access Token

1. Go to: https://github.com/settings/tokens/new
2. You'll be prompted to authenticate with your 2FA/authenticator
3. Enter your authenticator code
4. Fill in the form:
   - **Token name:** "MPNN Repository Push"
   - **Expiration:** 30 days (or longer if you prefer)
   - **Scopes:** Check only "repo" (full control of private repositories)
5. Click "Generate token"
6. **COPY the token immediately** (you can't see it again!)

## Step 2: Use Token for Push

When git prompts for a password, use:
- **Username:** Your GitHub username (e.g., AaravG42)
- **Password:** The token you just copied

## Step 3: Git Will Remember It

After the first push, git will store these credentials and you won't need to enter them again for future pushes.

---

## Why Not Direct Password?

GitHub disabled password authentication because:
- Tokens are more secure (limited scope)
- 2FA doesn't work with direct passwords over git
- Tokens can be revoked individually
- Better security practices

However, **tokens work WITH your authenticator** - you authenticate once via 2FA to generate the token, then use the token for git operations.

---

## Quick Summary

```bash
# 1. Go to https://github.com/settings/tokens/new
# 2. Authenticate with your authenticator app
# 3. Create token with "repo" scope
# 4. Copy the token
# 5. When prompted by git:
#    Username: AaravG42
#    Password: <paste token here>
```

That's it! Your 2FA/authenticator is used during token creation, then the token is used for all future pushes.
