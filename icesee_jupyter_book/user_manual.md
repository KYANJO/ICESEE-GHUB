cd icesee_jupyter_book

# 1) minimal landing page
cat > index.md <<'EOF'
# ICESEE-ONLINE

Welcome to the ICESEE-ONLINE documentation.
EOF

# 2) minimal MyST/Jupyter Book v2 config
cat > myst.yml <<'EOF'
version: 1
project:
  title: ICESEE-ONLINE

exports:
  - format: html
    template: book
    articles:
      - index.md
EOF