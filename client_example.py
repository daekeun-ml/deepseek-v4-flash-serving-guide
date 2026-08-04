"""DeepSeek-V4-Flash-0731 vLLM 서버용 OpenAI 호환 클라이언트 예제.

먼저 serve_deepseek_v4_flash.sh 로 서버를 띄운 뒤 실행.
    uv run --with openai client_example.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "deepseek-v4-flash"

messages = [{"role": "user", "content": "17 * 19 는 얼마야? 최종 정수만 답해."}]


def run(label, **extra_body):
    print(f"\n=== {label} ===")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body=extra_body or None,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    # Non-think — 빠른 응답
    run("non-think")

    # Think High — 명시적 CoT
    run(
        "think-high",
        chat_template_kwargs={"thinking": True, "reasoning_effort": "high"},
    )

    # Think Max — 최대 추론 (0731 변형은 max-model-len >= 393216 필요, 여긴 131072라 생략 가능)
    # run(
    #     "think-max",
    #     chat_template_kwargs={"thinking": True, "reasoning_effort": "max"},
    # )

    # Tool calling 예제
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "주어진 도시의 현재 날씨를 반환",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    print("\n=== tool-calling ===")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "서울 날씨 어때?"}],
        tools=tools,
    )
    print(resp.choices[0].message)
