### 분산 시스템 프레임 워크

uber workflow

분산시스템은 어떤 시스템이냐에 따라 다양한 요구사항이 있고, 그 요구사항에 맞는 시스템 개발을 각각해야한다.

![캡처](D:\git\MLops\캡처.PNG)

### Data Centric MLops

- data quality를 improve 한다.

-> consistency, error rate, diversity, coverage, feedback frequency, size ...

- good data, clean data, (**Tidy data**)

- 작지만 clean or consistency한 data의 중요성
- 궁극적 과정 : data build 하는 과정을 잘 만들고 다른 MLops 부분에서 좋은 feedback을 받아서 문제를 풀 수 있도록 도와주는 것 자체가 ML 문제를 해결하기 위한 본질적인 과정이라고 생각한다.
- 이정권님의 예시 ) 
  - 기간 :   <2달
  - 라벨링 구축 과정 인원 : <30명
  - 이미지 데이터 규모 : <20000장
  - 1 iteration의 규모임!
- data building 과정이 다른 부서에서는 black box처럼 여겨지는 경우가 있다.
- data의 개선을 통해서 ML을 개선해야 한다는 게 data centric 관점임.
- 현실적으로 : 실제 프로젝트 진행중에 데이터 구축 과정에서 쓸만한 모델이 완성되어있지 않은 경우도 많았고, 결국 모델이 서비스 되는 수준까지 가려면 다른 것들이 고려되어야 하는데 그러다보면 human intellegence를 뽑아내는 과정에서 feedback을 주기가 어렵다.
- 베이지안 딥러닝 관점 등,,,
- 내부적으로 전용 모델을 만들고 거기로부터 signal을 받는것도 말이될 수 있겠구나 
- Keep labels consistent ! ! ! ! ! ! ! ! ! !
- => NLP에서는 어떻게 구현할 수 있을까?
  - 1) 중복데이터셋에 대한 여러 작업자의 작업
  - 2) 오버뷰(sampling -> consistency measure -> 전체 consistency measure)
  - ========= 이건 언어 모델의 일관성에 대한, 언어 모델의 knowledge에 대한 글 ============== 
  - Measuring and Improving Consistency in Pretrained Language Models
  - Language Models as Knowledge Bases?

