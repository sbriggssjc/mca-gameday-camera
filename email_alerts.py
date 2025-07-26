import os
import smtplib
from email.message import EmailMessage
from typing import Iterable


def load_env() -> dict:
    env = {}
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                env.setdefault(key, val)
    env.update(os.environ)
    return env


def send_email(subject: str, body: str, recipients: Iterable[str]) -> None:
    env = load_env()
    smtp_user = env.get('SMTP_USER')
    smtp_pass = env.get('SMTP_PASS')
    if not smtp_user or not smtp_pass:
        print('Missing SMTP_USER/SMTP_PASS environment variables; email not sent.')
        return
    smtp_server = env.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(env.get('SMTP_PORT', '587'))

    msg = EmailMessage()
    msg['From'] = smtp_user
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
    except Exception as exc:
        print(f'Failed to send email: {exc}')
