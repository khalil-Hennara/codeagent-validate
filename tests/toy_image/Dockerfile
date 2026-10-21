# Toy task image for tests/test_generate_docker.py:  docker build -t toy/calc:1 tests/toy_image
# main = buggy base commit; branch `upstream` + tag v9.9 hold the "future fix" that prepare.py must hide.
FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq git patch > /dev/null && rm -rf /var/lib/apt/lists/*
COPY pytest /usr/bin/pytest
RUN mkdir -p /testbed /r2e_tests && cd /testbed && git init -q -b main \
 && printf 'def add(a, b):\n    return a + b\n' > calc.py && git add -A \
 && git -c user.email=a@b -c user.name=a commit -qm "add calc" \
 && printf 'def add(a, b):\n    return a - b\n' > calc.py \
 && git -c user.email=a@b -c user.name=a commit -qam "refactor (introduces bug)" \
 && git checkout -qb upstream && printf 'def add(a, b):\n    return a + b  # FUTURE FIX\n' > calc.py \
 && git -c user.email=a@b -c user.name=a commit -qam "future fix" && git tag v9.9 && git checkout -q main \
 && echo secret > /r2e_tests/test_hidden.py
# an editable-style install pointing at /testbed (prepare.py must repoint it to /workspace) and a
# conda-style path that contains "testbed" but must stay untouched
RUN SP=$(python3 -c 'import site; print(site.getsitepackages()[0])') && mkdir -p "$SP" \
 && printf '/testbed\n/opt/miniconda3/envs/testbed/lib/extra\n' > "$SP/toy_editable.pth"
WORKDIR /testbed
