# API 게이트웨이 구성

DeepSeek-V4-Flash-0731 vLLM 서버([README.md](../README.md)) 또는 SageMaker 엔드포인트([sagemaker/](../sagemaker/README.md)) 앞에 게이트웨이를 두는 방법입니다.

- vLLM 서버 자체 튜닝: [tuning.md](tuning.md)
- 실측 성능 근거: [benchmark.md](benchmark.md)

## 왜 필요한가

`vllm serve`가 노출하는 `:8000`을 그대로 쓰면 아래가 전부 빠집니다.

| 없는 것 | 결과 |
|---|---|
| 인증 | 포트에 닿는 누구나 155.4 GiB 모델을 무료로 사용 |
| 사용자별 쿼터 | 한 명이 슬롯을 전부 점유하면 나머지는 대기 |
| TLS | 프롬프트와 응답이 평문 전송 |
| 감사 로그 | 누가 얼마나 썼는지 알 수 없음 |
| 다중 백엔드 | 서버 1대가 죽으면 전체 장애 |

vLLM에도 `--api-key`가 있지만 **단일 공유 키**라서 사용자별 쿼터나 회수가 불가능해요. 그래서 별도 계층이 필요합니다.

## 타임아웃의 두 종류

게이트웨이를 고르기 전에 이것부터 알아야 합니다. LLM은 응답이 수십 초 걸리니까요.

| 타입 | 무엇을 재나 | 스트리밍으로 해결되나 |
|---|---|---|
| **유휴**(idle) | 마지막 데이터 이후 침묵한 시간 | **예**. 토큰이 올 때마다 리셋됨 |
| **전체**(total) | 요청 시작부터 끝까지 | 아니요 |

SSE 스트리밍을 쓰면 토큰이 계속 흐르므로 유휴 타임아웃은 사실상 무력화돼요. 실측으로 확인한 결과입니다. Nginx `proxy_read_timeout`을 **3초**로 줄여놓고 출력 400토큰을 요청했습니다.

| 방식 | 결과 | 소요 시간 |
|---|---|---|
| 비스트리밍 | **504 Gateway Time-out** | 3.01초에 끊김 |
| 스트리밍 | **200 OK** | **21.81초 (통과)** |

타임아웃이 3초인데 21.8초짜리 응답이 성공했습니다. TTFT가 62ms라 첫 청크가 3초 안에 오고, 이후 약 56ms 간격으로 계속 오니 3초 침묵이 발생할 일이 없기 때문이에요.

각 게이트웨이가 어느 타입인지가 선택을 결정합니다.

| 게이트웨이 | 타입 | 기본값 | 최대값 |
|---|---|---|---|
| LiteLLM `request_timeout` | 전체 | 6,000초 | 설정한 만큼 |
| **ALB** `idle_timeout.timeout_seconds` | 유휴 | 60초 | **4,000초** |
| **API Gateway** 통합 타임아웃 | **전체** | **29초** | 상향 요청 필요 |

**API Gateway만 전체 시간을 잽니다.** 그래서 스트리밍으로도 29초를 넘길 수 없고, LLM 앞에 두기 까다로워요.

## 방식 선택

| | LiteLLM | ALB | API Gateway |
|---|---|---|---|
| 정체 | 내가 띄우는 컨테이너 | AWS 관리형 | AWS 관리형 |
| 추가되는 것 | 가상 키, 팀별 예산과 쿼터, 모델 라우팅, 폴백 | TLS 종료, 다중 인스턴스 분산, 헬스체크 | IAM/Cognito 인증, 사용량 플랜, WAF |
| **토큰과 비용 인식** | **함** | 못 함 | 못 함 |
| SSE 스트리밍 | 통과 | 통과 | **제약 있음** |
| 타임아웃 | 전체(넉넉히 설정) | 유휴(최대 4,000초) | **전체 29초** |
| 비용 | 무료(오픈소스) | 시간 + 처리량 | 요청 건당 |
| 언제 | 팀별로 키와 예산을 나눠야 할 때 | LiteLLM 여러 대를 앞에서 분산할 때 | 조직이 IAM/WAF를 요구할 때 |

세 계층은 서로 대체재가 아니라 겹쳐 쓸 수 있어요. ALB나 API Gateway가 앞, LiteLLM이 뒤입니다.

### LiteLLM만 할 수 있는 것

LLM에서 결정적인 차이입니다.

```
요청 A: "안녕"                   12 토큰
요청 B: 100페이지 문서 요약      80,000 토큰
```

ALB와 API Gateway 입장에서는 둘 다 "요청 1건"이에요. 실제 비용은 6,000배 차이인데 똑같이 셉니다. LiteLLM은 응답의 `usage`를 읽으므로 "팀 A는 월 $100까지" 같은 제한이 가능해요.

### 단계별 구성

```
1단계  앱 → vLLM:8000                       로컬 실험
2단계  앱 → LiteLLM (키, 예산) → vLLM         사용자가 여러 명
3단계  앱 → ALB → LiteLLM ×2 → vLLM ×N       무중단 배포 / 다중화
```

필요해질 때 한 층씩 추가하는 게 맞습니다. 처음부터 여러 층을 쌓으면 문제가 생겼을 때 어느 층 때문인지 찾기 어려워져요.

LiteLLM 공식 문서는 프로덕션 구성으로 **ALB 또는 Kubernetes Ingress**를 권장합니다("services are stateless; run 2+ replicas behind a load balancer"). API Gateway는 권장 목록에 없어요.

### API Gateway는 언제 쓰나

기술적으로 더 좋아서가 아니라 **조직 제약 때문에** 쓰는 구성입니다.

- 회사 규정상 모든 외부 노출 API가 API Gateway를 경유해야 함
- AWS IAM이나 Cognito 계정 체계와 반드시 통합해야 함
- WAF가 필수 요구사항

자유롭게 고를 수 있다면 ALB가 LLM에 더 맞아요. 유휴 타임아웃이라 스트리밍이 그냥 통과하고, 4,000초까지 설정할 수 있습니다.

```bash
# ALB 유휴 타임아웃을 600초로 (기본 60초)
aws elbv2 modify-load-balancer-attributes \
  --load-balancer-arn "$ALB_ARN" \
  --attributes "Key=idle_timeout.timeout_seconds,Value=600"
```

---

## Option A: LiteLLM Proxy

가상 API 키, 팀별 예산과 쿼터, 모델 라우팅을 담당합니다. 클라이언트는 OpenAI SDK를 그대로 써요.

`litellm/config.yaml`:

```yaml
model_list:
  # hosted_vllm/ 접두사가 OpenAI 호환 vLLM 서버로 라우팅
  - model_name: deepseek-v4-flash            # 클라이언트가 부르는 이름
    litellm_params:
      model: hosted_vllm/deepseek-v4-flash   # --served-model-name과 일치해야 함
      api_base: http://vllm:8000/v1
      api_key: "none"                        # vLLM에 --api-key를 걸었으면 그 값
      rpm: 600
      tpm: 2000000
    model_info:
      # 131072로 서빙 중. 초과 요청은 백엔드까지 가지 않고 여기서 거절
      max_input_tokens: 131072
      max_output_tokens: 32768

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY   # 이 키로만 가상 키를 발급 가능
  database_url: os.environ/DATABASE_URL       # 키와 예산 저장 (없으면 재시작 시 소실)

litellm_settings:
  # S5 p99 28.7초 + think-high 여유를 감안
  request_timeout: 600
  # vLLM 자체 prefix caching과는 별개 계층
  cache: true
  cache_params:
    type: redis
    ttl: 600

router_settings:
  num_retries: 2
  allowed_fails: 3
  cooldown_time: 30
```

실행:

```bash
docker run -d --name litellm \
  -p 4000:4000 \
  -v "$PWD/litellm/config.yaml:/app/config.yaml" \
  -e LITELLM_MASTER_KEY="sk-master-CHANGE_ME" \
  -e DATABASE_URL="postgresql://user:pass@host:5432/litellm" \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml --port 4000
```

vLLM이 호스트에서 도는 경우 `api_base`를 `http://host.docker.internal:8000/v1`로 바꿉니다.

### 팀별 키 발급

```bash
# 팀 A: 월 100달러 상한, 분당 60요청
curl -s http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-master-CHANGE_ME" \
  -H "Content-Type: application/json" \
  -d '{"models":["deepseek-v4-flash"],
       "max_budget":100, "budget_duration":"30d",
       "rpm_limit":60, "tpm_limit":200000,
       "metadata":{"team":"team-a"}}'
```

두 종류의 키를 구분해야 해요.

| 키 | 역할 | 누가 가지나 |
|---|---|---|
| master key | 키를 **발급**할 권한 | 관리자만 |
| 가상 키(발급 결과) | 모델을 **사용**할 권한 | 각 팀 |

발급된 키로 호출하면 사용량이 자동 집계되고, 예산 초과 시 게이트웨이에서 거절됩니다.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-발급된키")
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "안녕"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

[client_example.py](../client_example.py)의 `base_url`만 4000번으로 바꾸면 non-think, think-high, tool-calling 모두 그대로 동작합니다(실측 확인).

### 여러 백엔드 묶기: 폴백과 로드밸런싱은 다르다

가장 많이 실수하는 지점입니다.

> 같은 `model_name`으로 여러 백엔드를 등록하면 **폴백이 아니라 로드밸런싱 풀**이 됩니다.

같은 이름에 로컬 vLLM과 SageMaker를 등록하고 호출해본 결과예요.

```
시도 1: vLLM으로 라우팅 (성공)
시도 2: SageMaker로 라우팅 (실패, 자격증명 없음)
시도 3~6: SageMaker로 라우팅 (실패)
```

의도는 "로컬 우선, 장애 시 SageMaker"였지만 실제로는 두 곳에 분산됐습니다. 내 GPU가 멀쩡한데 요청 절반이 유료 백엔드로 나가므로 비용이 예상 밖으로 발생하고, 백엔드별 품질 차이가 사용자에게 그대로 노출돼요.

**우선순위를 원하면 이름을 다르게 두고 `fallbacks`로 명시합니다.**

```yaml
model_list:
  - model_name: flash-local                    # 이름을 구분한다
    litellm_params:
      model: hosted_vllm/deepseek-v4-flash
      api_base: http://10.0.1.20:8000/v1
      api_key: "none"

  - model_name: flash-sagemaker
    litellm_params:
      model: sagemaker_chat/deepseek-v4-flash-endpoint
      aws_region_name: us-west-2

litellm_settings:
  # flash-local이 실패할 때만 flash-sagemaker로 넘어간다
  fallbacks: [{"flash-local": ["flash-sagemaker"]}]
```

같은 조건으로 검증한 결과입니다.

| 구성 | 내 GPU로 라우팅 | 유료 백엔드로 새어나감 |
|---|---|---|
| 같은 이름 (로드밸런싱) | 1 / 6 | **5 / 6** |
| **`fallbacks` 명시** | **10 / 10** | **0 / 10** |

| 목적 | 방법 |
|---|---|
| **폴백** (평소 A만, 장애 시 B) | 이름을 다르게 + `fallbacks` |
| **로드밸런싱** (A와 B에 분산) | 같은 이름 + `routing_strategy` |

같은 이름 방식은 **성능과 비용이 동등한 백엔드가 여러 대일 때만** 적절해요. 동일 스펙 vLLM 서버 3대를 묶는 경우가 그렇습니다.

### AWS 백엔드를 함께 묶을 때

Bedrock과 SageMaker 자체 호스팅은 이름이 비슷해도 성격이 정반대입니다.

| | Bedrock | SageMaker 자체 호스팅 |
|---|---|---|
| 모델 | AWS가 준비한 것 (Claude, Nova 등) | **내가 올린 것** |
| 과금 | **토큰당** | **인스턴스 시간당** (요청 0건이어도) |
| 유휴 시 | $0 | 계속 발생 |
| 성격 | 모델을 사는 것 | 내 서버를 AWS가 관리 |

SageMaker 접두사 선택도 주의가 필요해요.

| 접두사 | 스트리밍 |
|---|---|
| `sagemaker/` | **가짜**. 응답을 다 받은 뒤 조각내서 전달 |
| **`sagemaker_chat/`** | **진짜** 스트리밍, OpenAI 형식 |

가짜 스트리밍은 첫 조각이 응답 완료 후에 오므로 **유휴 타임아웃 우회 효과가 사라집니다.** vLLM을 SageMaker에 올렸다면 반드시 `sagemaker_chat/`을 쓰세요.

Bedrock까지 포함한 구성입니다.

```yaml
model_list:
  # 핵심 작업: 내 GPU (토큰 비용 없음)
  - model_name: flash-local
    litellm_params:
      model: hosted_vllm/deepseek-v4-flash
      api_base: http://10.0.1.20:8000/v1
      api_key: "none"

  # 재해 복구용: 내 GPU 전체 장애 시에만
  - model_name: flash-sagemaker
    litellm_params:
      model: sagemaker_chat/deepseek-v4-flash-endpoint
      aws_region_name: us-west-2

  # 가벼운 작업 전용: 유휴 시 $0이라 상시 대기에 유리
  - model_name: light-tasks
    litellm_params:
      model: bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0
      aws_region_name: us-west-2

litellm_settings:
  fallbacks: [{"flash-local": ["flash-sagemaker"]}]
```

묶는 것 자체가 목표가 아닙니다. **역할이 다른 백엔드를 각자 이름으로 두고 폴백만 명시하는 것**이 요점이에요.

| 묶는 게 타당한 경우 | 묶지 말아야 하는 경우 |
|---|---|
| 용도별로 다른 모델 (가벼운 작업은 Bedrock) | **같은 모델을 여러 곳에 두고 자동 분산** |
| 재해 복구 (전체 장애 시 폴백) | 데이터 외부 전송이 금지된 환경 |
| 트래픽 급증 흡수 (피크만 관리형으로) | 백엔드별 품질이 다른데 섞어 노출 |
| 여러 백엔드 지출을 한 곳에서 집계 | "옵션이 많으면 좋으니까" |

### SageMaker를 백엔드로 쓸 때

SageMaker에는 하드 제약이 두 개 있습니다.

| 제약 | 값 | 영향 |
|---|---|---|
| 컨테이너 응답 시간 | **60초** | 초과하면 SageMaker가 끊음. 긴 출력은 스트리밍 필수 |
| 요청과 응답 본문 | **6,291,456 바이트 (6 MB)** | 131K 컨텍스트를 꽉 채운 요청이 닿을 수 있음 |
| SDK 소켓 타임아웃 | 문서 권장 **70초** | 늘려두지 않으면 클라이언트가 먼저 끊음 |

또 SageMaker는 **SigV4 서명**을 요구하므로 OpenAI SDK가 직접 호출할 수 없어요. SigV4는 요청 본문과 시각으로 매 요청 새로 계산하는 서명이라 `api_key`에 미리 넣어둘 수가 없습니다. 다음 요청은 본문이 달라 서명이 무효가 되니까요.

| 방법 | 장점 | 단점 |
|---|---|---|
| boto3 직접 호출 | 게이트웨이 불필요 | 코드가 SageMaker 전용이 됨 |
| LiteLLM 경유 (`sagemaker_chat/`) | OpenAI SDK 그대로, 백엔드 교체 자유 | 컨테이너 1개 추가 |

이건 SageMaker를 쓸 때만 해당합니다. 로컬 vLLM만 쓴다면 vLLM 자체가 OpenAI 호환이라 서명이 필요 없어요.

```python
# 게이트웨이 없이 직접 호출할 때는 read_timeout을 반드시 늘린다
import boto3, json
from botocore.config import Config

rt = boto3.client("sagemaker-runtime", config=Config(read_timeout=70))
resp = rt.invoke_endpoint_with_response_stream(
    EndpointName="deepseek-v4-flash-endpoint",
    ContentType="application/json",
    Body=json.dumps({"model": "deepseek-v4-flash",
                     "messages": [{"role": "user", "content": "안녕"}],
                     "stream": True}),
)
for event in resp["Body"]:
    if "PayloadPart" in event:
        print(event["PayloadPart"]["Bytes"].decode(), end="", flush=True)
```

### LiteLLM은 어디에 배포하나

운영 환경에서는 게이트웨이와 모델을 다른 서버에 두는 것이 표준입니다.

```
[게이트웨이 계층]                      [모델 계층]
값싼 CPU 인스턴스                      GPU 서버
  - LiteLLM              ──내부망──▶     - vLLM만
  - Postgres, Redis
```

다만 흔히 생각하는 이유(자원 경쟁)는 실측하면 근거가 약해요.

| 항목 | LiteLLM 실측값 |
|---|---|
| GPU 사용 | **없음** |
| 메모리 | **648 MB** (499 GB 호스트의 0.13%) |
| CPU (8개 동시 요청 중) | 0.02% |

실제 분리 이유는 세 가지입니다.

**1) 재시작 비용 차이.** LiteLLM은 약 30초, vLLM은 **5~6분**(가중치 155.4 GiB 로딩 + CUDA 그래프 캡처)입니다. "모델 하나 추가" 같은 변경이 6분 재시작을 유발하는 상황을 만들지 않아야 해요.

**2) 확장 방향이 반대.** LiteLLM 부하는 CPU를, vLLM 부하는 GPU를 요구합니다. 게이트웨이 트래픽 때문에 비싼 GPU 서버를 늘리는 건 낭비예요.

**3) 여러 백엔드를 묶는 위치.** LiteLLM이 GPU 서버 A에 있으면 A가 죽을 때 서버 B나 SageMaker 폴백도 함께 못 씁니다.

**검증 단계에서는 같은 서버가 낫습니다.** 서버를 나누면 네트워크 설정이 변수로 추가되어 원인 파악이 어려워져요.

| 단계 | 배치 |
|---|---|
| 검증, 학습 | **같은 서버, 컨테이너로만 분리** |
| 소규모 운영 | 같은 서버로도 충분 (자원 0.13%) |
| 본격 운영 | 분리 |

별도 서버로 옮길 때는 `api_base`만 사설 IP로 바꾸고, vLLM은 사설망에만 노출합니다.

### LiteLLM을 2대 이상 둘 때

게이트웨이를 이중화해도 **뒤쪽 vLLM이 1대면 그것이 단일 장애점**으로 남습니다. LiteLLM을 죽였을 때 vLLM은 정상(`200`)인데 서비스는 중단(`000`)되는 것을 확인했어요. 반대도 같습니다.

```
LiteLLM ×2  →  vLLM ×1  →  GPU 2장
   (이중화)      (단일!)     ← 여기가 끊기면 앞단 이중화는 무의미
```

이 저장소 구성에서 vLLM을 2대로 늘리려면 가중치 155.4 GiB × 2 = **GPU 4장**이 필요합니다.

| 상황 | 2대 이상이 필요한가 |
|---|---|
| GPU 2장, vLLM 1대 | **아니요**. 뒤가 단일이라 소용 없음 |
| **무중단 배포가 필요** | 예. 설정 변경 때 멈추지 않음 |
| vLLM 서버가 여러 대 | 예. 게이트웨이가 병목이자 단일 장애점이 됨 |
| 게이트웨이 CPU가 포화 | 예. 부하 분산 목적 |

실무에서 가장 흔한 동기는 **무중단 배포**입니다. LiteLLM은 무상태이므로 키와 예산을 Postgres, 캐시를 Redis에 두면 여러 대로 늘릴 수 있어요.

---

## Option B: AWS API Gateway

조직 제약으로 API Gateway가 필수인 경우입니다. 두 구성이 가능하고 **스트리밍 여부가 갈림길**이에요.

### B-1. VPC Link 직결 (비스트리밍, 짧은 출력 전용)

```
Client → API Gateway (REST, Regional) → VPC Link → NLB → EC2:8000 (vLLM)
```

Lambda가 없어 가장 단순하지만 **29초 통합 타임아웃**이 그대로 적용되고 응답이 버퍼링됩니다. [benchmark.md](benchmark.md)의 실측값을 대입하면 문제가 드러나요.

| 시나리오 | E2EL p50 | E2EL p99 | 29초 안에 들어오나 |
|---|---|---|---|
| S1. 단일 사용자 (출력 200) | 1.9초 | 1.9초 | 여유 있음 |
| S2. 일반 대화 c=8 | 3.6초 | 3.7초 | 여유 있음 |
| S3. 높은 부하 c=32 | 7.2초 | **17.0초** | 아슬아슬 |
| S4. 문서 요약 c=8 (입력 4K) | 5.2초 | **15.3초** | 아슬아슬 |
| S5. 코드 생성 c=8 (출력 1,000) | **18.7초** | **28.7초** | **한계선** |

출력 1,000토큰짜리 요청은 비스트리밍으로 통과시킬 수 없습니다. think-high 추론은 더 나빠요. 이 구성을 쓴다면 **`max_tokens`로 출력을 강제 제한**해야 안전합니다.

```bash
# NLB 타깃을 vLLM 인스턴스로 등록해둔 상태에서 VPC Link 생성
aws apigatewayv2 create-vpc-link \
  --name vllm-link \
  --subnet-ids subnet-aaa subnet-bbb \
  --security-group-ids sg-xxx

# REST API에 private integration 연결 (Regional 타입이어야 타임아웃 상향 가능)
aws apigateway put-integration \
  --rest-api-id "$API_ID" --resource-id "$RESOURCE_ID" \
  --http-method POST --type HTTP_PROXY --integration-http-method POST \
  --uri "http://internal-nlb-xxx.elb.us-west-2.amazonaws.com/v1/chat/completions" \
  --connection-type VPC_LINK --connection-id "$VPC_LINK_ID" \
  --timeout-in-millis 29000
```

`--timeout-in-millis`를 29000보다 크게 주려면 Service Quotas에서 `L-E5AE38E3` 상향을 받아야 합니다. **Regional 또는 private REST API에서만** 가능하고 HTTP API는 대상이 아니에요.

### B-2. Lambda STREAM 모드 (스트리밍)

```
Client → API Gateway (REST, Lambda proxy, responseTransferMode=STREAM)
       → Lambda (InvokeWithResponseStream) → vLLM:8000 또는 SageMaker
```

API Gateway가 응답을 스트리밍하는 유일한 경로입니다. Lambda 출력이 **메타데이터 JSON → 8바이트 null 구분자 → 페이로드** 순서를 지켜야 하고, 구분자는 스트림 첫 16 KB 안에 나와야 해요. 형식이 틀리면 Lambda를 호출하고도 500을 반환합니다.

응답 스트리밍은 **Node.js 관리형 런타임**에서 지원되고, Python은 커스텀 런타임이나 Lambda Web Adapter가 필요합니다.

`lambda/stream_proxy.mjs`:

```javascript
// API Gateway responseTransferMode=STREAM 용 프록시.
// vLLM의 SSE를 그대로 클라이언트로 흘려보낸다.
const VLLM_URL = process.env.VLLM_URL;   // http://10.0.1.20:8000/v1/chat/completions
const DELIMITER = Buffer.alloc(8);       // 8바이트 null. API Gateway 규약

export const handler = awslambda.streamifyResponse(async (event, responseStream) => {
  const upstream = await fetch(VLLM_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: event.body,
  });

  // 1) 메타데이터 JSON
  responseStream.write(JSON.stringify({
    statusCode: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Transfer-Encoding": "chunked",
    },
  }));

  // 2) 구분자. 반드시 첫 16KB 안에
  responseStream.write(DELIMITER);

  // 3) 페이로드를 청크 단위로 그대로 전달
  const reader = upstream.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      responseStream.write(value);
    }
  } finally {
    responseStream.end();
  }
});
```

통합 생성 시 URI의 API 버전(`2021-11-15`)과 액션(`response-streaming-invocations`)이 일반 proxy와 다릅니다.

```bash
aws apigateway put-integration \
  --rest-api-id "$API_ID" --resource-id "$RESOURCE_ID" \
  --http-method POST --type AWS_PROXY --integration-http-method POST \
  --uri "arn:aws:apigateway:us-west-2:lambda:path/2021-11-15/functions/arn:aws:lambda:us-west-2:$ACCOUNT:function:vllm-stream-proxy/response-streaming-invocations"
```

주의할 점입니다.

| 항목 | 내용 |
|---|---|
| Lambda 타임아웃 | 최대 15분. 단 **클라이언트가 끊어도 중단되지 않고 전체 실행 시간이 과금** |
| 대역폭 | 첫 6 MB 무제한, 이후 2 MBps. 토큰 스트림은 거의 닿지 않음 |
| VPC | 사설 IP의 vLLM에 닿으려면 Lambda를 VPC에 넣어야 함. Function URL은 VPC 안에서 스트리밍 미지원 |
| 비용 | 18초 생성이면 18초 전액 과금 |

### 29초 대응 방법

| 방법 | 내용 | 언제 |
|---|---|---|
| **1. 게이트웨이를 바꾼다** | ALB로 교체 | 조직 제약이 없다면 이게 정답 |
| 2. Lambda STREAM 모드 | 위 B-2 구성 | API Gateway가 필수일 때 |
| 3. 쿼터 상향 | `L-E5AE38E3` (Regional/private만) | 계정 스로틀 축소가 요구될 수 있음 |
| 4. `max_tokens` 상한 | 최악의 응답 시간을 예측 가능하게 | 항상 함께 적용 권장 |
| 5. 비동기 전환 | job ID 반환 후 나중에 조회 | 리포트 생성 등 배치성 작업만 |

사용량 플랜으로 키별 쿼터를 붙일 수 있습니다.

```bash
aws apigateway create-usage-plan \
  --name vllm-standard \
  --throttle burstLimit=20,rateLimit=10 \
  --quota limit=10000,period=DAY \
  --api-stages "apiId=$API_ID,stage=prod"
```

---

## 권장 조합

**① 팀별로 키와 예산을 나눠야 할 때**

```
Client → LiteLLM:4000 → vLLM:8000                 (flash-local)
                      └→ SageMaker / Bedrock       (fallbacks로 명시)
```

대부분 여기서 끝납니다. LiteLLM이 TLS도 처리할 수 있어요.

**② 무중단 배포 또는 다중화**

```
Client → ALB (TLS, 헬스체크 /health/readiness, idle_timeout 600s)
       → LiteLLM ×2 이상 → vLLM ×N
```

**③ 조직이 API Gateway를 요구하는 경우**

```
Client → API Gateway (IAM/Cognito, WAF, 사용량 플랜)
       → Lambda STREAM → LiteLLM 또는 vLLM
```

29초 제약과 Lambda 규약을 감당해야 하고 Lambda 실행 시간 전체가 과금됩니다. 자유롭게 고를 수 있다면 ②를 쓰세요.

## 붙인 뒤 확인할 것

게이트웨이는 성능을 **떨어뜨리기만** 합니다. 얼마나 떨어뜨렸는지 측정하세요.

```bash
# 직결
vllm bench serve --backend openai-chat --base-url http://localhost:8000 \
  --model deepseek-v4-flash --dataset-name random \
  --random-input-len 204 --random-output-len 200 \
  --max-concurrency 8 --num-prompts 80 \
  --save-result --result-filename direct.json

# 게이트웨이 경유
vllm bench serve --backend openai-chat --base-url http://localhost:4000 \
  --model deepseek-v4-flash --dataset-name random \
  --random-input-len 204 --random-output-len 200 \
  --max-concurrency 8 --num-prompts 80 \
  --save-result --result-filename via_gateway.json
```

기준선은 [benchmark.md](benchmark.md)의 S2입니다(TTFT p50 210ms, TPOT p50 17.1ms).

| 증상 | 원인 |
|---|---|
| 첫 토큰이 응답 완료 시점에 한꺼번에 도착 | 응답 버퍼링. 업스트림의 `X-Accel-Buffering` 헤더도 확인 |
| 긴 출력에서만 504 | 게이트웨이 타임아웃이 생성 시간보다 짧음. 가장 흔한 원인 |
| 429가 백엔드 여유와 무관하게 발생 | 게이트웨이 레이트 제한이 실제 처리량보다 낮음 |
| TTFT가 직결 대비 수백 ms 증가 | TLS 핸드셰이크 반복 또는 업스트림 keepalive 미적용 |
| 요청이 의도한 백엔드로 안 감 | 같은 `model_name` 중복 등록. `fallbacks`로 전환 |

## 참고 링크

- [ALB 속성 설정](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/edit-load-balancer-attributes.html) 유휴 타임아웃 1~4,000초
- [API Gateway 쿼터](https://docs.aws.amazon.com/general/latest/gr/apigateway.html) 통합 타임아웃 기본 29,000 ms, `L-E5AE38E3`로 상향
- [API Gateway Lambda 응답 스트리밍](https://docs.aws.amazon.com/apigateway/latest/developerguide/response-transfer-mode-lambda.html) STREAM 모드 출력 형식
- [Lambda 응답 스트리밍](https://docs.aws.amazon.com/lambda/latest/dg/configuration-response-streaming.html) 대역폭 제한, VPC 제약
- [SageMaker InvokeEndpoint](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html) 60초 응답 제한, 6 MB 페이로드
- LiteLLM 프로바이더 문서: [Proxy 설정](https://docs.litellm.ai/docs/proxy/configs), [vLLM](https://docs.litellm.ai/docs/providers/vllm), [Bedrock](https://docs.litellm.ai/docs/providers/bedrock), [SageMaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)
- [tuning.md](tuning.md) vLLM 서버 최적화
- [benchmark.md](benchmark.md) 인용한 실측값
