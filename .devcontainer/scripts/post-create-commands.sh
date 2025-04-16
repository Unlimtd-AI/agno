# Check for required environment variables
: "${GIT_USER_EMAIL:?Need to set GIT_USER_EMAIL}"
: "${GIT_USER_NAME:?Need to set GIT_USER_NAME}"


sudo chown -R $USER:root /usr/local/pip-global

# Git config
chmod 600 $HOME/.ssh/git/id_ed25519_github
eval "$(ssh-agent -s)"
ssh-add $HOME/.ssh/git/id_ed25519_github

# Add to .bashrc
echo "chmod 600 \$HOME/.ssh/git/id_ed25519_github" >> ~/.bashrc
echo "eval \$(ssh-agent -s)" >> ~/.bashrc
echo "ssh-add \$HOME/.ssh/git/id_ed25519_github" >> ~/.bashrc

git config --global --add safe.directory "$(pwd)"

git config --global user.email "$GIT_USER_EMAIL" &&
git config --global user.name "$GIT_USER_NAME"

# Install Rust and uv for both root and vscode user
curl -Ls https://astral.sh/uv/install.sh | bash && \
   sudo ln -s "$HOME/.local/bin/uv" /usr/local/bin/uv

curl -fsSL https://ollama.com/install.sh | sh

# Add to PATH for both root and vscode user
PATH="$HOME/.local/bin:${PATH}"

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && \
   echo 'export PATH="$PWD/.venv/bin:$PATH"' >> ~/.bashrc && \
   source ~/.bashrc

