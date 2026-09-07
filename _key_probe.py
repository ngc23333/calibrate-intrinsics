# -*- coding: utf-8 -*-
"""按键探针：确认 OpenCV 窗口能否收到空格/回车。用完可删。

运行：python _key_probe.py
在弹出的窗口上按键，终端会打印键码；按 q 退出。
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("窗口已弹出，请点一下窗口标题栏让它获得焦点，然后按 空格 / 回车 / d 试试")

last = -1
while True:
    ok, frame = cap.read()
    if not ok:
        frame = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(frame, "last key=%d" % last, (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("key probe", frame)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        last = key
        print("收到键码:", key)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
