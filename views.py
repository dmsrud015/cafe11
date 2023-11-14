from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators import gzip
from django.shortcuts import render

from . models import Customer
########################################################################
# 라이브러리 불러오기
########################################################################
import torch
import cv2
import numpy as np
from datetime import datetime
import pymysql

# DeepSort Engine 라이브러리
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes, check_imshow, xyxy2xywh, increment_path
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
# Create your views here.


# 전역변수 선언
g_start_time = None
g_end_time = None


def index(request):
  
  out_count = 0 # 테이크
  in_count = 0  # 매장

  # 현재 날짜
  current_data = datetime.now().date()
  print("current_data --  시간 확인용 -- ", current_data )
  # 날짜에 맞는 데이터
  customers_data = Customer.objects.filter(cusdate = current_data)
  print(customers_data, "확인용입니다.")

  for customer in customers_data:
      total_time = customer.custotaltime.split(":")

      hours = int(total_time[0])
      minutes = int(total_time[1])

      # 시간 + 분 
      total_minutes = (hours * 60) + minutes

      if total_minutes < 10:
          out_count += 1 
      else:
          in_count += 1

  # 영상시간계산
  if g_start_time is not None and g_end_time is not None:
      video_duration = g_end_time - g_start_time
      video_duration_seconds = video_duration.total_seconds()
      print("video_duration_seconds = ", video_duration_seconds)
  else:
      video_duration_seconds = 0
  



  rtn_msg = {
      'customers' : customers_data,
      'takeout' : out_count,
      'dinein' : in_count,
      'current_time' : current_data,
      'video_duration_seconds' : video_duration_seconds
  }


  return render(request, 'cafemanage.html', rtn_msg) 


###################################################
# 인공지능 처리 부분
# #######################3#########################
# yolov5 설정
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]
model.conf = 0.80

#  yolov5 DeepSort Engine
device = select_device('')  # '0'=GPU, ''=CPU
cfg = get_config()
cfg.merge_from_file('./deep_sort/configs/deep_sort.yaml')
deepsort = DeepSort('osnet_x0_25', max_dist=cfg.DEEPSORT.MAX_DIST, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=False)

# 전역 변수
g_customerList = {}

# 객체가 인식될때 색상(Color)과 이름(Names) 정보를 저장
#  hasattr object의 속성 존재를 확인 하는 함수 
names = model.module.names if hasattr(model, 'module') else model.names
print('names = ', names)

# 카페 영상파일
mov_url = "media/mov/1.mp4"

# 카페 영상파일 플레이
cap = cv2.VideoCapture(mov_url)

def dicAdd(ids, cus_code, cus_yolo):
    global g_customerList

    if ids in g_customerList:
        # 키가 존재하면 수정
        now = datetime.now()
        valueData = g_customerList.get(ids, '')

        if valueData != '':
            # 총 value데이터를 값을 의미합니다
            #예를들어 print 할시 값은은 ['00035', 94.25, '2023-08-28 16:27:27', '2023-08-28 16:27:32', '0:00:05', '2023-08-28', '16:27:32'] 이렇게 나옵니다 
            # print("value 데이터입니다.",valueData)

            #now_yolo는 총 value 데이터 값 의 첫번째를 출력하는 코드입니다.
            now_yolo = valueData[1]
            cus_start = valueData[2]
            # print("now_yolo[2]입니다:",cus_start)
            if now_yolo > cus_yolo:
                chg_yolo = now_yolo
            else:
                chg_yolo = cus_yolo
            # print("chg_yolo입니다:   ",chg_yolo)
            cur_end = now.strftime('%Y-%m-%d %H:%M:%S')
            cur_date = now.strftime('%Y-%m-%d')
            cur_time = now.strftime('%H:%M:%S')

            # 시간차이 계산
            dStart = datetime.strptime(cus_start, "%Y-%m-%d %H:%M:%S")
            dEnd = datetime.strptime(cur_end, "%Y-%m-%d %H:%M:%S")

            diff = str(dEnd - dStart)

        g_customerList[ids] = [cus_code, chg_yolo, cus_start, cur_end, diff, cur_date, cur_time]
    else:
        # 존재하지 않으면 추가
        now = datetime.now()
        cur_now = now.strftime('%Y-%m-%d %H:%M:%S')
        cur_date = now.strftime('%Y-%m-%d')
        cur_time = now.strftime('%H:%M:%S')

        g_customerList[ids] = [cus_code, cus_yolo, cur_now, cur_now, 0, cur_date, cur_time]

def dicView():
    global g_customerList

    print('-' * 50)
    for key, value in g_customerList.items():
        print(key, value)
    print('=' * 50)

def dicSaveDB():
    # # MySQL 연결
    # conn = pymysql.connect(host='localhost', user='cafemanage', password='1234',
    #                     db='cafedb', charset='utf8')
    
    # # Connection 으로부터 Cursor 생성
    # curs = conn.cursor()

    # 데이터 저장
    for key, value in g_customerList.items():
        dicData = value
        # [cus_code, chg_yolo, cus_start, cur_end, diff, cur_date, cur_time]
        cus_code    = dicData[0]
        chg_yolo    = dicData[1]
        cus_start   = dicData[2]
        cus_end     = dicData[3]
        diff        = dicData[4]
        cur_date    = dicData[5]
        cur_time    = dicData[6]

        # # SQL문 실행
        # sql = "select max(cusid) from customer"
        # curs.execute(sql)

        idx = 0

        # # 데이타 Fetch
        # max_result = curs.fetchall()
        # # print(max_result[0][0])
    
        if max_result[0][0] == None:
            idx = 1
        else:
            for data in max_result:
                idx = data[0] + 1
                print('idx = ', idx)

        # DB 저장
        data = Customer()
        data.cuscode = cus_code
        data.cusyolo = chg_yolo
        data.cusstarttime = cus_start
        data.cusendtime = cus_end
        data.custotaltime = diff
        data.cusdate = cur_date
        data.cuscreate = cur_time
        data.save()

      


        # 입력 테스트
    #     sql = "INSERT INTO customer (cusid, cuscode, cusyolo, cusstarttime, cusendtime, custotaltime, cusdate, cuscreate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    #     val = (idx, cus_code, chg_yolo, cus_start, cus_end, diff, cur_date, cur_time)
    #     curs.execute(sql, val)

    #     conn.commit()

    # # SQL문 실행
    # sql = "select * from customer"
    # curs.execute(sql)
    
    # # 데이타 Fetch
    # rows = curs.fetchall()
    # print(rows)     # 전체 rows

    
    # # Connection 닫기
    # conn.close()

def gen(camera):
    
  global g_start_time, g_end_time

  while True:
    ret, frame = camera.read()

    # 영상 시작 시간
    if g_start_time is None:
        g_start_time = datetime.now()

    if not ret:
        
        # 영상 종료 시간
        g_end_time = datetime.now()
        # 내용 저장
        dicSaveDB()
        break
    
        # 영상의 크기 조정한다.
    frame_resize = cv2.resize(frame, dsize=(800, 480), interpolation=cv2.INTER_LINEAR)

######################################################
# Yolov5 DeepSort 작업영역
######################################################
    result = model(frame_resize, augment=True)
        # 인식한 정보 상세보기
    annotator = Annotator(frame_resize, line_width=2, pil=False)

# 인식한 정보의 존재하면 처리 루틴
    pre_result = result.pred[0]
    if pre_result is not None and len(pre_result):
        xywh = xyxy2xywh(pre_result[:,0:4])
        xywh_numpy = np.array(xywh)
        confs = pre_result[:,4]
        class_num = pre_result[:,5]

        output = deepsort.update(xywh.cpu(), confs.cpu(), class_num.cpu(), frame_resize)

        if len(output) > 0:
            for idx, (d_output, d_confs) in enumerate(zip(output, confs)):
                bbox = d_output[0:4]
                ids = d_output[4]
                class_num = d_output[5]
                class_names = names[int(class_num)]
                conf_num = round(float(d_confs) * 100, 2)
                name_info = f'[NO:{ids}] {class_names} {conf_num}%'

                dicAdd(ids, '00035', conf_num)
                dicView()

                annotator.box_label(bbox, name_info, color=colors(int(class_num), True))

  

    ret, jpeg = cv2.imencode('.jpg', frame_resize)
    frame = jpeg.tobytes()
    yield (b'--frame\r\n'
          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        # return frame
        # cv2.imshow('video', frame_resize)

def video_feed(request):
    return StreamingHttpResponse(gen(cap), content_type="multipart/x-mixed-replace;boundary=frame")