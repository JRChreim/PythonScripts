# TECConcursos automation

The scripts use Python 3.10 or newer, PyYAML, and Playwright with Chromium.
Create the virtual environment separately on each computer: a virtual
environment copied between Ubuntu and Windows is not portable.

## Windows (PowerShell)

From the repository root, create and prepare the environment once:

```powershell
py -3 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r .\TECConcursos\requirements.txt
.\.venv-local\Scripts\python.exe -m playwright install chromium
```

If `.venv-local` already exists and its Python executable works, only the last
two commands are needed.

Set the credentials for the current PowerShell session and run the script:

```powershell
$env:TEC_EMAIL = "your-email@example.com"
$env:TEC_PASSWORD = "your-password"

.\.venv-local\Scripts\python.exe .\TECConcursos\gerar_caderno_tec.py --list-cadernos
.\.venv-local\Scripts\python.exe .\TECConcursos\gerar_caderno_tec.py --caderno "Contabilidade - Multibanca"
```

## Ubuntu (Bash)

From the repository root, create and prepare the environment once:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r ./TECConcursos/requirements.txt
./.venv/bin/python -m playwright install chromium
```

If Chromium reports missing Linux system libraries, install them once with:

```bash
sudo ./.venv/bin/python -m playwright install-deps chromium
```

Set the credentials for the current shell and run the script:

```bash
export TEC_EMAIL="your-email@example.com"
export TEC_PASSWORD="your-password"

./.venv/bin/python ./TECConcursos/gerar_caderno_tec.py --list-cadernos
./.venv/bin/python ./TECConcursos/gerar_caderno_tec.py --caderno "Contabilidade - Multibanca"
```

`gerar_caderno_tec.py` performs the configured automation. `tec_macro.py` is
the older interactive/manual-login helper and can be run with the same Python
executable. YAML paths are resolved relative to the scripts, so both entry
points can be launched from the repository root or another working directory.

Credentials are read from environment variables and are never meant to be
stored in the repository. The repository `.gitignore` excludes local virtual
environments, caches, and `.env` files.
