from tools import calculator

def test_calculator():
    assert calculator("2+3")=="5"


def test_calculator_complex():
    # 测试: (15 + 27) * 3
    assert calculator("(15 + 27) * 3") == "126"
    # 测试: 2 ** 3 + 5
    assert calculator("2 ** 3 + 5") == "13"

def test_execute_tool_unknown():
    from tools import execute_tool
    result = execute_tool("unknown",{})
    assert "未找到" in result

def test_execute_tool_wrong_params():
    from tools import execute_tool
    result = execute_tool("calculator", {"wrong_param": "10"})
    # 你要写：断言 result 里有"失败"字样
    assert "失败" in result