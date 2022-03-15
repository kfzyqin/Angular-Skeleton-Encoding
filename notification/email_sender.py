import smtplib

from notification.email_templates import *
from .email_config import *
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .html_templates import *


def send_email(receivers, email_type, msg_content):
    return
    template_1 = None
    sub_email = None
    if email_type == 'exp_end':
        template_1 = exp_complete_1
        template_2 = exp_complete_2
        sub_email = 'Watchtower at Warrumbul: Experiment Complete'
    elif email_type == 'test_end':
        template_1 = exp_progress_1
        template_2 = exp_progress_2
        sub_email = 'Watchtower at Warrumbul: Experiment Progress'
    elif email_type == 'error':
        template_1 = emergency_1
        template_2 = emergency_2
        sub_email = 'Watchtower at Warrumbul: Emergency'
    else:
        raise NotImplementedError

    for a_person in receivers:

        def send_email(msg):
            try:
                server = smtplib.SMTP('smtp-relay.sendinblue.com', port=587)
                server.ehlo()
                server.starttls()
                server.login(EMAIL_S_ADDRESS, PASSWORD)
                message = msg
                server.sendmail(EMAIL_S_ADDRESS, a_person, message)
                server.quit()
            except Exception:
                traceback.print_exc()

        message = MIMEMultipart("alternative")
        if sub_email is not None:
            message["Subject"] = sub_email
        else:
            message["Subject"] = "实验进展: 加急公文 御赐金牌 马上飞递"
        message["From"] = EMAIL_S_ADDRESS
        message["To"] = a_person

        html = template_1 + msg_content + template_2

        part2 = MIMEText(html, 'html', 'utf-8')

        # message.attach(part1)
        message.attach(part2)

        send_email(message.as_string())

        print(f'Success: Email sent to {a_person}')


if __name__ == '__main__':
    test_msg = '这是一个测试. '
    send_email(receivers=['zhenyue.qin@anu.edu.au'],
               email_type='error',
               msg_content=test_msg)