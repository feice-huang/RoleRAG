import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generate.anti2qa import anti2qa
from data_generate.chat2qa import chat2qa
from data_generate.conv2summary import conv2summary
from data_generate.cot2all import cot2all
from data_generate.qa2all import qa2all
from data_generate.qa2cot import qa2cot
from data_generate.qa2anticot import qa2anticot
from data_generate.statement2qa import statement2qa
from data_generate.summary2qa import summary2qa
from data_generate.wiki2statement import wiki2statement

# world = "家有儿女"
# role = "夏东海"
apikey = "sk-MA7hKS37UdRUmP3Xz4BzHt3Rqj6QFbRoEagxcmFwwBBHyZR6"

"""
wiki -> statement -> qa_statement(75) 
conversations -> summary -> qa_summary(25)
GPT -> qa_chat(25)
GPT -> qa_anti(75)
"""

world = "家有儿女"
role = "刘星"

# wiki2statement(role, apikey, "gpt-4o-mini")
# statement2qa(world, role, apikey, "gpt-4o")
# conv2summary(world, [role], apikey, model_name="gpt-4o-mini")
# summary2qa(world, role, apikey, "gpt-4o")
# chat2qa(world, role, apikey, "gpt-4o")
# anti2qa(world, role, apikey, "gpt-4o")
qa2all(world, role)
# qa2cot(world, role, apikey, "gpt-4o")
# qa2anticot(world, role, apikey, "gpt-4o")
# cot2all(world, role)


# world = "家有儿女"
# role = "小雨"

# wiki2statement(role, apikey, "gpt-4o-mini")
# statement2qa(world, role, apikey, "gpt-4o")
# conv2summary(world, [role], apikey, model_name="gpt-4o-mini")
# summary2qa(world, role, apikey, "gpt-4o")
# chat2qa(world, role, apikey, "gpt-4o")
# anti2qa(world, role, apikey, "gpt-4o")
# qa2all(world, role)
# qa2cot(world, role, apikey, "gpt-4o")
# qa2anticot(world, role, apikey, "gpt-4o")
# cot2all(world, role)


# world = "家有儿女"
# role = "小雪"

# wiki2statement(role, apikey, "gpt-4o-mini")
# statement2qa(world, role, apikey, "gpt-4o")
# conv2summary(world, [role], apikey, model_name="gpt-4o-mini")
# summary2qa(world, role, apikey, "gpt-4o")
# chat2qa(world, role, apikey, "gpt-4o")
# anti2qa(world, role, apikey, "gpt-4o")
# qa2all(world, role)
# qa2cot(world, role, apikey, "gpt-4o")
# qa2anticot(world, role, apikey, "gpt-4o")
# cot2all(world, role)