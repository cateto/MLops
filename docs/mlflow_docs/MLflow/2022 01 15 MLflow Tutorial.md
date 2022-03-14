# 2022.01.15 MLflow Tutorial

[GitHub - vhrehfdl/MLflow_tutorial](https://github.com/vhrehfdl/MLflow_tutorial)

<aside>
💡 git clone [https://github.com/vhrehfdl/MLflow_tutorial](https://github.com/vhrehfdl/MLflow_tutorial)

</aside>

### 인터파크 톡집사 ML팀

- 회사 내에 모델 Serving , Devops 팀은 따로 있다 ㅠ
- 모델 개발 외에도 너무 관리할 것들이 많았음..!!!
- MLflow는 설치가 놀라울 정도로 쉬웠음.
- Kubeflow는 구축이 어려웠음,,,,
- wandb : 데이터 학습과정에서 과정들을 기록해주는 툴 but 요금정책이 변화하면서 사용 그만둠.
- wandb는 유료 mlflow는 무료~

### MLflow를 구축하게 된 배경

- 모델 개발 과정에서 실험을 반복하며 발생하는 accuracy, f1 score, train loss, parameter 등을 기록해야 한다.
- 보통 엑셀이나 자기만의 기록하는 곳에 기록했을 것이다.
- 만약 실험이 100번 넘게 이루어지면....
- 자동화하는 방법은 없을까? 예쁜 UI로 기록을 모아서 볼 수 있을까?

### 실습

- 만약 서버 컴퓨터에서 띄우고 있다면 아래와 같은 명령어로 실행

```jsx
mlflow ui -h 0.0.0.0 -p 1010
```

- BentoML - django 에 비유,, mlflow - flask에 비유 ,,할수 있다 ㅎㅎㅎ
- 패키징화 해서 제공함. epoch , batch size, max length 같은것 (관리포인트가 조금 줄어듦)

- airflow는 데이터베이스에 데이터가 많이 쌓여있고 특정 벌크를 돌릴 때 사용함.

    ex) 고객이 만명이면 0~1000번까지 벌크 돌리는 데 효과적