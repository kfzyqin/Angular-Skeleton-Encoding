from notification.email_sender import send_email


test_msg = '这是一个测试. '
send_email(receivers=['zhenyue.qin@anu.edu.au'],
           email_type='exp_end',
           msg_content=test_msg)
