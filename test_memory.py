def test_memory_add_and_recall(tmp_path):
    from memory import Memory
    m = Memory(str(tmp_path / "test.json"))
    m.add("用户叫小明")
    m.add("用户喜欢跑步")
    results = m.recall("小明")
    assert "用户叫小明" in results

def test_memory_dedup(tmp_path):
    from memory import Memory
    m = Memory(str(tmp_path / "test.json"))
    m.add("用户叫小明")
    m.add("用户叫小明")   # 重复
    m.add("用户叫小明")   # 再重复
    assert len(m.facts) == 1   # 应该是多少？


def test_memory_persistence(tmp_path):
    from memory import Memory
    path = str(tmp_path / "test.json")

    # 第一个实例：写入
    m1 = Memory(path)
    m1.add("用户叫小明")

    # 第二个实例：新建，读同一文件
    m2 = Memory(path)
    results = m2.recall("小明")

    # 你要写：断言 results 里有"用户叫小明"
    assert "用户叫小明" in results