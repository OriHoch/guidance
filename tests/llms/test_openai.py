import guidance
import tiktoken
from ..utils import get_openai_llm

def assert_usage(model_name, prompt, expected_usage, expected_usage_usd):
    llm = guidance.llms.OpenAI(model_name, caching=True)
    llm.cache.clear()
    assert llm.get_usage_cost_usd() == 0
    guidance.Program(prompt, llm=llm)()
    assert dict(llm.usage) == expected_usage
    assert llm.get_usage_cost_usd() == expected_usage_usd
    assert dict(llm.usage_cached) == {}
    guidance.Program(prompt, llm=llm)()
    assert dict(llm.usage) == expected_usage
    assert dict(llm.usage_cached) == expected_usage
    assert llm.get_usage_cost_usd(llm.usage_cached) == llm.get_usage_cost_usd() == expected_usage_usd

def assert_dynamic_max_tokens(model_name, prompt, expected_test_data):
    guidance.llm = get_openai_llm(model_name)
    enc = tiktoken.encoding_for_model(guidance.llm.model_name)
    test_data = {}

    def max_tokens_callback_test(model_max_tokens, prompt_num_tokens):
        test_data.update(model_max_tokens=model_max_tokens, prompt_num_tokens=prompt_num_tokens)
        return 2

    guidance.llm.register_max_tokens_callback('test', max_tokens_callback_test)
    out = guidance(prompt)()
    assert len(enc.encode(out["res"])) == 2
    assert test_data == expected_test_data

def test_geneach_chat_gpt():
    """ Test a geneach loop with ChatGPT.
    """

    guidance.llm = get_openai_llm("gpt-3.5-turbo")

    chat_loop = guidance('''
    {{#system~}}
    You are a helpful assistant
    {{~/system}}

    {{~#geneach 'conversation' stop=False}}
    {{#user~}}
    This is great!
    {{~/user}}

    {{#assistant~}}
    {{gen 'this.response' temperature=0 max_tokens=3}}
    {{~/assistant}}
    {{#if (> @index 0)}}{{break}}{{/if}}
    {{~/geneach}}''')

    out = chat_loop()
    assert len(out["conversation"]) == 2

def test_usage():
    """ Test the usage tracking of OpenAI
    """

    assert_usage(
        "text-davinci-003",
        '''Hello, {{gen 'response' temperature=0 max_tokens=3}}''',
        {'completion_tokens': 3, 'prompt_tokens': 3, 'total_tokens': 6},
        0.00011999999999999999
    )
    assert_usage(
        "gpt-3.5-turbo",
        '''
        {{#system~}}
        You are a helpful assistant
        {{~/system}}
        {{#user~}}
        Hello,
        {{~/user}}
        {{#assistant~}}
        {{gen 'response' temperature=0 max_tokens=3}}
        {{~/assistant}}
        ''',
        {'completion_tokens': 3, 'prompt_tokens': 25, 'total_tokens': 28},
        5.6000000000000006e-05
    )


def test_dynamic_max_tokens():
    """ Test OpenAI generating a response with dynamic max tokens.
    """
    assert_dynamic_max_tokens(
        "text-davinci-003",
        '''
        Short story about cars:
        {{gen 'res' max_tokens_callback='test'}}
        ''',
        {'model_max_tokens': 4097, 'prompt_num_tokens': 9}
    )
    assert_dynamic_max_tokens(
        "gpt-3.5-turbo",
        '''
        {{#system~}}
        You are a helpful assistant
        {{~/system}}
        {{#user~}}
        Hello, please write a short story
        {{~/user}}
        {{#assistant~}}
        {{gen 'res' max_tokens_callback='test'}}
        {{~/assistant}}
        ''',
        {'model_max_tokens': 4096, 'prompt_num_tokens': 30}
    )
