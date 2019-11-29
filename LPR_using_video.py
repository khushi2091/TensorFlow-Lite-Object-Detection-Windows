import time
import base64
import requests
import cv2
import re
import json

def detect_license_plate(image):
    license_number = ""
    url = 'https://vision.googleapis.com/v1/images:annotate?key=AIzaSyA1GYvKPFVjzk-xPgVkA73pPoNtG7ksRxA'
    res = ''
    image = cv2.imencode('.jpg', image)[1].tostring()
    img_base64 = base64.b64encode(image)
    ig = str(img_base64)
    ik=ig.replace('b\'','')
    headers={'content-type': 'application/json'}
    
    data ="""{
      "requests": [
        {
          "image": {
                   "content": '"""+ik[:-1]+"""'
                
                    },
          
          "features": [
            {
              "type": "TEXT_DETECTION"
            }
          ]
        }
      ]
    }"""
    r = requests.post(url, headers=headers,data=data)
    result = json.loads(r.text)
    try:
        result = result['responses'][0]['textAnnotations'][0]['description']
    except:
        return(res)
    result = result.replace('\n', '')
    result = re.sub('\W+','', result)
    mystates = ['AP','AR','AS','BR','CG','GA','GJ','HR' ,'HP' ,'JK','JH','KA','KL','MP','MH','MN','ML','MZ','NL' ,'OD','PB' ,'RJ','SK','TN','TS','TR','UA','UK','UP','WB','AN','CH','DN','DD','DL' ,'LD','PY']
    
    if(len(result) > 0):
        for word in mystates:
            if(word in result):
                res = re.findall(word + "[0-9]{1,2}\s*[A-Z]{1,4}\s*[0-9]{1,4}\s*]?", result)
                if(len(res) >0):
                    res = res
                    break
    if(len(res) > 0):
        license_number = res[0]
    else:
        license_number = ''
    return(license_number)

count = -1
cap = cv2.VideoCapture(1)
license_p = ''
msg = ''
cv2.namedWindow('License Plate', cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

time.sleep(2.0)
while(True):
    count += 1
    if(len(license_p) < 1):
        msg = ''
    lpr = "License Plate : " + str(license_p)
    #lp = str(license_p)
    rv, frame = cap.read()
    if(rv != True):
        break
    if(count % 10 == 0 ):
        license_p = detect_license_plate(frame)
        if(len(license_p) > 0):
            lpr = "License Plate : " + str(license_p)
                
    cv2.rectangle(frame ,(5 , 8 ),(400, 50), (0,0 ,0), 2)
    cv2.putText(frame, lpr , (10, 35), font, 0.8, (0,0, 255), 1, cv2.LINE_AA)
    cv2.imshow('License Plate', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
