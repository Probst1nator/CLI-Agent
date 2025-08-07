
# These imports are always available
from agent_utils import SearchWeb

# Agents are instantiable objects
webSearcher = SearchWeb()

# print type and docstring of the webSearcher
print(type(webSearcher))
print(webSearcher.__doc__)

# # Agents have a .run(..kwargs) method that takes specific args and returns a specific type
# webSearchSummary = webSearcher.run([f"{datetime.datetime.now().strftime('%Y-%m-%d')} weather Bavaria"])

# textToBool = TextToBoolAgent()
# isHot = textToBool.run(webSearchSummary, "Is it currently hotter than 20°C in Bavaria?")