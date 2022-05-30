 MLops의 목표

- Production 환경, 배포하는 과정에서 Research의 모델이 재현 가능해야 함.
- (= 현실의 Risk있는 환경에서 잘 버틸 수 있어야 함)
- 빠른 시간 내에 가장 적은 위험을 부담하며 아이디어 단계부터 프러덕션까지 ML Project를 진행할 수 있도록 기술적 마찰을 줄이는 것

Serving : Production(Real World) 환경에 사용할 수 있도록 모델을 배포

 -  Serving 방식
    	-  1) Batch 단위로 여러 데이터를 한번에 예측
    	-  2) API 형태로 요청이 올때마다 예측
    	-  Serving환경의 의존성 : 파이썬 버전, 라이브러리 등...
 -  Batch Serving
    	-  1) Airflow, Cronjob으로 스케쥴링 작업
    	-  학습 / 예측을 별도의 작업으로 설정
    	-  학습 : 1주일에 1번
    	-  예측 : 10분, 30분, 1시간에 1번씩
 -  Online Serving(API 형태)
     -  Lv 1. Flask , Fast API
     -  Lv 2. Lv1 + docker
     -  Lv 3. Lv2 + Kubernetes
     -  Lv 4. Serving 프레임웍
         -  ex) Kubeflow, BentoML... 등
     -  모델 많아질경우에는 서빙 프레임웍 사용
 -  처음부터 API 형태로 serving해야하는 것은아님.
 -  최초엔 Batch Serving을 구축, 결과를 DB에 저장하고, 서버는 그 데이터를 주기적으로 가져가는 방식으로 통신
 -  점점 운영하면서 API 형태로 변환
 -  