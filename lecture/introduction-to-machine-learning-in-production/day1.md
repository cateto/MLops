# INTRO

**"change"**

Entire Life Cycle

hands on skills

X : pictures of phones

Y : defective or not

challenges ahead to get a valuable production deployment running.

"concept drift or data drift"



1. Steps of an ML Project

   1) Scoping : Define Project

   2) Data

   3) Modeling (highly iterrative task)

   4) Deployment

![](C:\Users\som\Desktop\git\MLops\lecture\introduction-to-machine-learning-in-production\캡처1.PNG)

2. Case study: speech recognition

"conventions specific transcription has happened to use for an audio clip"

데이터 일관성이 중요하다.

"systematic frameworks for making sure you have high quality data."

체계적인 프레임웍이 하이퀄리티의 데이터를 갖도록함.

=======

# Deployment

### Concept drift and Data drift

모델 deploy 이후 데이터나 콘셉의 변경

ex) covid19 이전의 온라인구매와 이후 온라인 구매

주택 가격의 변화- 정치가나, 셀러브리티의 언급량

### Software Engineering issues

1. realtime 예측 or batch 예측
2. cloud로 돌리냐 엣지 컴퓨팅이냐
3. compute resources(CPU/GPU/memory)

4. realtime 애플리케이션에서는 latency랑 throughput (QPS)도 중요

   ex) 1000 queries /second

5. Logging

6. security and privacy

#### Common Deployment cases

1. new product/capability
2. automate/assist with manual task
3. replace previous ML system
4. KEY IDEAS!!!
   * Gradual ramp up with monitoring
   * Rollback

