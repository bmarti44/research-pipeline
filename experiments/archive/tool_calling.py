"""
Tool Calling Scenarios for Two-Stage Experiment

100 tool scenarios (20 per tool type × 5 tool types) + 30 control scenarios.
Each scenario has an ambiguity level: EXPLICIT, IMPLICIT, or CONTROL.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolScenario:
    """A single tool calling scenario."""
    id: str
    query: str                           # User message
    expected_tool: Optional[str]         # None for controls
    expected_params: dict[str, str]      # Expected parameters
    ambiguity: str                       # EXPLICIT, IMPLICIT, CONTROL
    domain: str                          # Tool type
    notes: str = ""                      # Optional notes


# =============================================================================
# MEMORY_SAVE Scenarios (20 total: 10 EXPLICIT, 10 IMPLICIT)
# =============================================================================

MEMORY_SAVE_SCENARIOS = [
    # EXPLICIT (10)
    ToolScenario(
        id="mem_exp_001",
        query="Save this: the API key is sk-abc123xyz",
        expected_tool="memory_save",
        expected_params={"key": "api_key", "value": "sk-abc123xyz"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_002",
        query="Remember my database password: supersecret99",
        expected_tool="memory_save",
        expected_params={"key": "database_password", "value": "supersecret99"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_003",
        query="Store this for later - the connection string is postgres://user:pass@host:5432/db",
        expected_tool="memory_save",
        expected_params={"key": "connection_string", "value": "postgres://user:pass@host:5432/db"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_004",
        query="Please save my preferred timezone: America/New_York",
        expected_tool="memory_save",
        expected_params={"key": "timezone", "value": "America/New_York"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_005",
        query="I want you to remember that my project name is 'Apollo'",
        expected_tool="memory_save",
        expected_params={"key": "project_name", "value": "Apollo"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_006",
        query="Save my AWS region preference: us-west-2",
        expected_tool="memory_save",
        expected_params={"key": "aws_region", "value": "us-west-2"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_007",
        query="Store the webhook URL: https://hooks.slack.com/services/T00/B00/xxx",
        expected_tool="memory_save",
        expected_params={"key": "webhook_url", "value": "https://hooks.slack.com/services/T00/B00/xxx"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_008",
        query="Remember this: my team uses Python 3.11",
        expected_tool="memory_save",
        expected_params={"key": "python_version", "value": "3.11"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_009",
        query="Save that our deploy target is kubernetes",
        expected_tool="memory_save",
        expected_params={"key": "deploy_target", "value": "kubernetes"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_exp_010",
        query="Please remember my email: dev@example.com",
        expected_tool="memory_save",
        expected_params={"key": "email", "value": "dev@example.com"},
        ambiguity="EXPLICIT",
        domain="memory_save",
    ),

    # IMPLICIT (10)
    ToolScenario(
        id="mem_imp_001",
        query="I keep forgetting the connection string. It's postgres://prod:secret@db.example.com:5432/main",
        expected_tool="memory_save",
        expected_params={"key": "connection_string", "value": "postgres://prod:secret@db.example.com:5432/main"},
        ambiguity="IMPLICIT",
        domain="memory_save",
        notes="User expresses difficulty remembering - implies they want it saved",
    ),
    ToolScenario(
        id="mem_imp_002",
        query="The OAuth client ID is client_12345, I always have to look it up",
        expected_tool="memory_save",
        expected_params={"key": "oauth_client_id", "value": "client_12345"},
        ambiguity="IMPLICIT",
        domain="memory_save",
        notes="'Always have to look it up' suggests desire to save",
    ),
    ToolScenario(
        id="mem_imp_003",
        query="Just so you know, we use the staging endpoint https://api.staging.example.com for testing",
        expected_tool="memory_save",
        expected_params={"key": "staging_endpoint", "value": "https://api.staging.example.com"},
        ambiguity="IMPLICIT",
        domain="memory_save",
        notes="Sharing information 'just so you know' implies persistence",
    ),
    ToolScenario(
        id="mem_imp_004",
        query="FYI, the S3 bucket name is my-company-assets-prod",
        expected_tool="memory_save",
        expected_params={"key": "s3_bucket", "value": "my-company-assets-prod"},
        ambiguity="IMPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_imp_005",
        query="The service account email is sa-123@project.iam.gserviceaccount.com, you'll probably need this later",
        expected_tool="memory_save",
        expected_params={"key": "service_account", "value": "sa-123@project.iam.gserviceaccount.com"},
        ambiguity="IMPLICIT",
        domain="memory_save",
        notes="'Need this later' explicitly suggests persistence",
    ),
    ToolScenario(
        id="mem_imp_006",
        query="Oh, one thing - our default branch is 'main' not 'master'",
        expected_tool="memory_save",
        expected_params={"key": "default_branch", "value": "main"},
        ambiguity="IMPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_imp_007",
        query="BTW the repo is at github.com/myorg/myrepo",
        expected_tool="memory_save",
        expected_params={"key": "repo_url", "value": "github.com/myorg/myrepo"},
        ambiguity="IMPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_imp_008",
        query="Note: max file upload size is 10MB in our system",
        expected_tool="memory_save",
        expected_params={"key": "max_upload_size", "value": "10MB"},
        ambiguity="IMPLICIT",
        domain="memory_save",
        notes="'Note:' prefix suggests something to remember",
    ),
    ToolScenario(
        id="mem_imp_009",
        query="The feature flag for dark mode is DARK_MODE_ENABLED, I never remember which one it is",
        expected_tool="memory_save",
        expected_params={"key": "dark_mode_flag", "value": "DARK_MODE_ENABLED"},
        ambiguity="IMPLICIT",
        domain="memory_save",
    ),
    ToolScenario(
        id="mem_imp_010",
        query="Our company uses 4-space indentation, not tabs - mentioning because we discussed this before",
        expected_tool="memory_save",
        expected_params={"key": "indentation", "value": "4 spaces"},
        ambiguity="IMPLICIT",
        domain="memory_save",
    ),
]


# =============================================================================
# WEB_SEARCH Scenarios (20 total: 10 EXPLICIT, 10 IMPLICIT)
# =============================================================================

WEB_SEARCH_SCENARIOS = [
    # EXPLICIT (10)
    ToolScenario(
        id="web_exp_001",
        query="Search for the latest Python 3.12 release notes",
        expected_tool="web_search",
        expected_params={"query": "Python 3.12 release notes"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_002",
        query="Look up the current AWS Lambda memory limits",
        expected_tool="web_search",
        expected_params={"query": "AWS Lambda memory limits"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_003",
        query="Can you search for React 19 new features?",
        expected_tool="web_search",
        expected_params={"query": "React 19 new features"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_004",
        query="Find information about the GitHub Copilot pricing",
        expected_tool="web_search",
        expected_params={"query": "GitHub Copilot pricing"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_005",
        query="Search the web for Kubernetes 1.30 changelog",
        expected_tool="web_search",
        expected_params={"query": "Kubernetes 1.30 changelog"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_006",
        query="Google the error message 'ECONNREFUSED' for Node.js",
        expected_tool="web_search",
        expected_params={"query": "ECONNREFUSED Node.js error"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_007",
        query="Look up how to configure nginx reverse proxy",
        expected_tool="web_search",
        expected_params={"query": "nginx reverse proxy configuration"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_008",
        query="Search for PostgreSQL 16 performance improvements",
        expected_tool="web_search",
        expected_params={"query": "PostgreSQL 16 performance improvements"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_009",
        query="Find the official Docker documentation for multi-stage builds",
        expected_tool="web_search",
        expected_params={"query": "Docker multi-stage builds documentation"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_exp_010",
        query="Search for TypeScript 5.4 release announcement",
        expected_tool="web_search",
        expected_params={"query": "TypeScript 5.4 release announcement"},
        ambiguity="EXPLICIT",
        domain="web_search",
    ),

    # IMPLICIT (10)
    ToolScenario(
        id="web_imp_001",
        query="What's the current version of Next.js?",
        expected_tool="web_search",
        expected_params={"query": "current Next.js version"},
        ambiguity="IMPLICIT",
        domain="web_search",
        notes="Asking about 'current' version requires up-to-date info",
    ),
    ToolScenario(
        id="web_imp_002",
        query="Is Vercel still offering free tier for hobby projects?",
        expected_tool="web_search",
        expected_params={"query": "Vercel free tier hobby projects"},
        ambiguity="IMPLICIT",
        domain="web_search",
        notes="'Still offering' implies need for current information",
    ),
    ToolScenario(
        id="web_imp_003",
        query="I'm getting a weird SSL certificate error in production today",
        expected_tool="web_search",
        expected_params={"query": "SSL certificate error production"},
        ambiguity="IMPLICIT",
        domain="web_search",
        notes="Recent issue might be service outage - need to check",
    ),
    ToolScenario(
        id="web_imp_004",
        query="Did they fix the Safari flexbox bug in the latest version?",
        expected_tool="web_search",
        expected_params={"query": "Safari flexbox bug fixed latest version"},
        ambiguity="IMPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_imp_005",
        query="How much does Azure OpenAI cost per token nowadays?",
        expected_tool="web_search",
        expected_params={"query": "Azure OpenAI cost per token pricing"},
        ambiguity="IMPLICIT",
        domain="web_search",
        notes="'Nowadays' implies need for current pricing",
    ),
    ToolScenario(
        id="web_imp_006",
        query="Anyone else having issues with npm registry today?",
        expected_tool="web_search",
        expected_params={"query": "npm registry issues outage today"},
        ambiguity="IMPLICIT",
        domain="web_search",
        notes="Checking for current service status",
    ),
    ToolScenario(
        id="web_imp_007",
        query="What's the recommended way to handle authentication in Next.js 14?",
        expected_tool="web_search",
        expected_params={"query": "Next.js 14 authentication recommended"},
        ambiguity="IMPLICIT",
        domain="web_search",
        notes="Best practices change with versions",
    ),
    ToolScenario(
        id="web_imp_008",
        query="Has anyone benchmarked Bun vs Node for API servers recently?",
        expected_tool="web_search",
        expected_params={"query": "Bun vs Node API server benchmark recent"},
        ambiguity="IMPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_imp_009",
        query="I heard there was a security vulnerability in log4j - is it patched?",
        expected_tool="web_search",
        expected_params={"query": "log4j security vulnerability patch status"},
        ambiguity="IMPLICIT",
        domain="web_search",
    ),
    ToolScenario(
        id="web_imp_010",
        query="What time does AWS re:Invent start this year?",
        expected_tool="web_search",
        expected_params={"query": "AWS re:Invent schedule this year"},
        ambiguity="IMPLICIT",
        domain="web_search",
        notes="Event timing requires current info",
    ),
]


# =============================================================================
# CODE_EXECUTE Scenarios (20 total: 10 EXPLICIT, 10 IMPLICIT)
# =============================================================================

CODE_EXECUTE_SCENARIOS = [
    # EXPLICIT (10)
    ToolScenario(
        id="code_exp_001",
        query="Run this Python code: print(sum(range(100)))",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "print(sum(range(100)))"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_002",
        query="Execute: const x = [1,2,3].map(n => n*2); console.log(x)",
        expected_tool="code_execute",
        expected_params={"language": "javascript", "code": "const x = [1,2,3].map(n => n*2); console.log(x)"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_003",
        query="Can you run `ls -la` and show me the output?",
        expected_tool="code_execute",
        expected_params={"language": "bash", "code": "ls -la"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_004",
        query="Please execute this: import sys; print(sys.version)",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "import sys; print(sys.version)"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_005",
        query="Run the following JavaScript: JSON.stringify({a: 1, b: 2}, null, 2)",
        expected_tool="code_execute",
        expected_params={"language": "javascript", "code": "JSON.stringify({a: 1, b: 2}, null, 2)"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_006",
        query="Execute this bash command: echo $PATH",
        expected_tool="code_execute",
        expected_params={"language": "bash", "code": "echo $PATH"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_007",
        query="Run: [x**2 for x in range(10)]",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "[x**2 for x in range(10)]"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_008",
        query="Please run this code: Array.from({length: 5}, (_, i) => i + 1)",
        expected_tool="code_execute",
        expected_params={"language": "javascript", "code": "Array.from({length: 5}, (_, i) => i + 1)"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_009",
        query="Execute: date +%Y-%m-%d",
        expected_tool="code_execute",
        expected_params={"language": "bash", "code": "date +%Y-%m-%d"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_exp_010",
        query="Run this Python snippet: from collections import Counter; print(Counter('mississippi'))",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "from collections import Counter; print(Counter('mississippi'))"},
        ambiguity="EXPLICIT",
        domain="code_execute",
    ),

    # IMPLICIT (10)
    ToolScenario(
        id="code_imp_001",
        query="What does 2**16 evaluate to?",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "2**16"},
        ambiguity="IMPLICIT",
        domain="code_execute",
        notes="Could be calculated, but running code is more reliable",
    ),
    ToolScenario(
        id="code_imp_002",
        query="I'm curious what sorted(['banana', 'apple', 'cherry']) returns",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "sorted(['banana', 'apple', 'cherry'])"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_003",
        query="Does this regex match? /^\\d{3}-\\d{4}$/.test('123-4567')",
        expected_tool="code_execute",
        expected_params={"language": "javascript", "code": "/^\\d{3}-\\d{4}$/.test('123-4567')"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_004",
        query="How many items are in this list after deduplication? [1, 2, 2, 3, 3, 3, 4]",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "len(set([1, 2, 2, 3, 3, 3, 4]))"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_005",
        query="What's the output of joining ['a', 'b', 'c'] with '-'?",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "'-'.join(['a', 'b', 'c'])"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_006",
        query="If I have {a: 1, b: 2} and spread {...obj, c: 3}, what's the result?",
        expected_tool="code_execute",
        expected_params={"language": "javascript", "code": "const obj = {a: 1, b: 2}; console.log({...obj, c: 3})"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_007",
        query="Hmm, I wonder if 'hello'.replace('l', 'L') replaces all or just first",
        expected_tool="code_execute",
        expected_params={"language": "javascript", "code": "'hello'.replace('l', 'L')"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_008",
        query="Does list(dict.fromkeys([1,2,2,3])) preserve order?",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "list(dict.fromkeys([1,2,2,3]))"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_009",
        query="What happens when you add null + 1 in JavaScript?",
        expected_tool="code_execute",
        expected_params={"language": "javascript", "code": "null + 1"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
    ToolScenario(
        id="code_imp_010",
        query="I need to verify the output of divmod(17, 5)",
        expected_tool="code_execute",
        expected_params={"language": "python", "code": "divmod(17, 5)"},
        ambiguity="IMPLICIT",
        domain="code_execute",
    ),
]


# =============================================================================
# FILE_OPERATION Scenarios (20 total: 10 EXPLICIT, 10 IMPLICIT)
# =============================================================================

FILE_OPERATION_SCENARIOS = [
    # EXPLICIT (10)
    ToolScenario(
        id="file_exp_001",
        query="Read the file config.json",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "config.json"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_002",
        query="Write 'Hello World' to output.txt",
        expected_tool="file_operation",
        expected_params={"operation": "write", "path": "output.txt", "content": "Hello World"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_003",
        query="Can you open README.md and show me what's in it?",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "README.md"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_004",
        query="Save this JSON to data.json: {\"name\": \"test\", \"value\": 42}",
        expected_tool="file_operation",
        expected_params={"operation": "write", "path": "data.json", "content": '{"name": "test", "value": 42}'},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_005",
        query="Read the contents of .env",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": ".env"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_006",
        query="Write the following to notes.txt: Meeting at 3pm",
        expected_tool="file_operation",
        expected_params={"operation": "write", "path": "notes.txt", "content": "Meeting at 3pm"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_007",
        query="Please read src/index.ts",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "src/index.ts"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_008",
        query="Create a file called test.py with: print('test')",
        expected_tool="file_operation",
        expected_params={"operation": "write", "path": "test.py", "content": "print('test')"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_009",
        query="Show me the package.json file",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "package.json"},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_exp_010",
        query="Write 'export const API_URL = \"https://api.example.com\"' to constants.js",
        expected_tool="file_operation",
        expected_params={"operation": "write", "path": "constants.js", "content": 'export const API_URL = "https://api.example.com"'},
        ambiguity="EXPLICIT",
        domain="file_operation",
    ),

    # IMPLICIT (10)
    ToolScenario(
        id="file_imp_001",
        query="What's in the package.json?",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "package.json"},
        ambiguity="IMPLICIT",
        domain="file_operation",
        notes="Asking about file contents implies reading",
    ),
    ToolScenario(
        id="file_imp_002",
        query="I need to check the Dockerfile",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "Dockerfile"},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_imp_003",
        query="Can you see what dependencies we have in requirements.txt?",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "requirements.txt"},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_imp_004",
        query="Let's look at the tsconfig.json settings",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "tsconfig.json"},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_imp_005",
        query="I wonder if the .gitignore includes node_modules",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": ".gitignore"},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_imp_006",
        query="Here's the updated config, it should go in app.config.js:\nmodule.exports = { port: 3000 }",
        expected_tool="file_operation",
        expected_params={"operation": "write", "path": "app.config.js", "content": "module.exports = { port: 3000 }"},
        ambiguity="IMPLICIT",
        domain="file_operation",
        notes="'Should go in' implies writing",
    ),
    ToolScenario(
        id="file_imp_007",
        query="Let me see the workflow definition in .github/workflows/ci.yml",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": ".github/workflows/ci.yml"},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_imp_008",
        query="The new environment variable should be in .env.example: DATABASE_URL=",
        expected_tool="file_operation",
        expected_params={"operation": "write", "path": ".env.example", "content": "DATABASE_URL="},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_imp_009",
        query="Does the Makefile have a build target?",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "Makefile"},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
    ToolScenario(
        id="file_imp_010",
        query="I want to see how the main function is structured in main.go",
        expected_tool="file_operation",
        expected_params={"operation": "read", "path": "main.go"},
        ambiguity="IMPLICIT",
        domain="file_operation",
    ),
]


# =============================================================================
# API_CALL Scenarios (20 total: 10 EXPLICIT, 10 IMPLICIT)
# =============================================================================

API_CALL_SCENARIOS = [
    # EXPLICIT (10)
    ToolScenario(
        id="api_exp_001",
        query="Make a GET request to https://api.github.com/users/octocat",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.github.com/users/octocat"},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_002",
        query="POST to https://httpbin.org/post with body {\"test\": true}",
        expected_tool="api_call",
        expected_params={"method": "POST", "url": "https://httpbin.org/post", "body": '{"test": true}'},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_003",
        query="Call the API endpoint https://jsonplaceholder.typicode.com/todos/1",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://jsonplaceholder.typicode.com/todos/1"},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_004",
        query="Send a DELETE request to https://api.example.com/items/123",
        expected_tool="api_call",
        expected_params={"method": "DELETE", "url": "https://api.example.com/items/123"},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_005",
        query="Make an API call to https://api.stripe.com/v1/customers with GET",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.stripe.com/v1/customers"},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_006",
        query="PUT this data to https://api.example.com/users/1: {\"name\": \"John\"}",
        expected_tool="api_call",
        expected_params={"method": "PUT", "url": "https://api.example.com/users/1", "body": '{"name": "John"}'},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_007",
        query="Hit the endpoint https://api.openweathermap.org/data/2.5/weather?q=London",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.openweathermap.org/data/2.5/weather?q=London"},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_008",
        query="Send a POST request to https://hooks.slack.com/services/xxx with {\"text\": \"Hello\"}",
        expected_tool="api_call",
        expected_params={"method": "POST", "url": "https://hooks.slack.com/services/xxx", "body": '{"text": "Hello"}'},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_009",
        query="Fetch data from https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_exp_010",
        query="Make a request to https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY"},
        ambiguity="EXPLICIT",
        domain="api_call",
    ),

    # IMPLICIT (10)
    ToolScenario(
        id="api_imp_001",
        query="Check if the health endpoint at https://api.example.com/health is responding",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.example.com/health"},
        ambiguity="IMPLICIT",
        domain="api_call",
        notes="'Check if responding' implies making a request",
    ),
    ToolScenario(
        id="api_imp_002",
        query="Is the API at https://api.myservice.com/v1/status up?",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.myservice.com/v1/status"},
        ambiguity="IMPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_imp_003",
        query="Can you see what https://jsonplaceholder.typicode.com/users returns?",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://jsonplaceholder.typicode.com/users"},
        ambiguity="IMPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_imp_004",
        query="I need to notify Slack when this is done - the webhook is https://hooks.slack.com/services/T00/B00/xxx",
        expected_tool="api_call",
        expected_params={"method": "POST", "url": "https://hooks.slack.com/services/T00/B00/xxx"},
        ambiguity="IMPLICIT",
        domain="api_call",
        notes="'Notify Slack' via webhook implies POST request",
    ),
    ToolScenario(
        id="api_imp_005",
        query="What's the current Bitcoin price from CoinGecko?",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"},
        ambiguity="IMPLICIT",
        domain="api_call",
        notes="Asking for live data implies API call",
    ),
    ToolScenario(
        id="api_imp_006",
        query="Let's test if the new endpoint https://api.myapp.com/v2/users works",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.myapp.com/v2/users"},
        ambiguity="IMPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_imp_007",
        query="Ping the server at https://api.example.com/ping",
        expected_tool="api_call",
        expected_params={"method": "GET", "url": "https://api.example.com/ping"},
        ambiguity="IMPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_imp_008",
        query="Create a new todo item via https://jsonplaceholder.typicode.com/todos - title should be 'Test'",
        expected_tool="api_call",
        expected_params={"method": "POST", "url": "https://jsonplaceholder.typicode.com/todos", "body": '{"title": "Test"}'},
        ambiguity="IMPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_imp_009",
        query="Update the user settings at https://api.example.com/settings with {\"theme\": \"dark\"}",
        expected_tool="api_call",
        expected_params={"method": "PUT", "url": "https://api.example.com/settings", "body": '{"theme": "dark"}'},
        ambiguity="IMPLICIT",
        domain="api_call",
    ),
    ToolScenario(
        id="api_imp_010",
        query="Remove the item from the cart - endpoint is https://api.shop.com/cart/item/456",
        expected_tool="api_call",
        expected_params={"method": "DELETE", "url": "https://api.shop.com/cart/item/456"},
        ambiguity="IMPLICIT",
        domain="api_call",
        notes="'Remove' implies DELETE request",
    ),
]


# =============================================================================
# CONTROL Scenarios (30 total - no tool should be called)
# =============================================================================

CONTROL_SCENARIOS = [
    ToolScenario(
        id="ctrl_001",
        query="What's the difference between REST and GraphQL?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_002",
        query="Can you explain what a closure is in JavaScript?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_003",
        query="How does async/await work?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_004",
        query="What are the SOLID principles?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_005",
        query="Explain the difference between SQL and NoSQL databases",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_006",
        query="What is dependency injection?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_007",
        query="How do I structure a React application?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_008",
        query="What's the best way to handle errors in Python?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_009",
        query="Can you describe the MVC pattern?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_010",
        query="What is the virtual DOM?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_011",
        query="How do microservices communicate?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_012",
        query="What's the difference between let and const?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_013",
        query="Explain how Git branching works",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_014",
        query="What is a race condition?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_015",
        query="How do I choose between useState and useReducer?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_016",
        query="What's the purpose of TypeScript interfaces vs types?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_017",
        query="Can you explain event-driven architecture?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_018",
        query="What is memoization and when should I use it?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_019",
        query="How do I prevent SQL injection?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_020",
        query="What's the difference between authentication and authorization?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_021",
        query="Explain the observer pattern",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_022",
        query="What is a deadlock and how do I avoid it?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_023",
        query="How does garbage collection work in JavaScript?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_024",
        query="What's the difference between process and thread?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_025",
        query="Explain CAP theorem",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_026",
        query="What is a webhook?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_027",
        query="How do I design a rate limiter?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_028",
        query="What's the singleton pattern good for?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_029",
        query="Explain the difference between PUT and PATCH",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
    ToolScenario(
        id="ctrl_030",
        query="What is eventual consistency?",
        expected_tool=None,
        expected_params={},
        ambiguity="CONTROL",
        domain="general",
    ),
]


# =============================================================================
# All Scenarios Combined
# =============================================================================

ALL_TOOL_SCENARIOS: list[ToolScenario] = (
    MEMORY_SAVE_SCENARIOS +
    WEB_SEARCH_SCENARIOS +
    CODE_EXECUTE_SCENARIOS +
    FILE_OPERATION_SCENARIOS +
    API_CALL_SCENARIOS
)

ALL_CONTROL_SCENARIOS: list[ToolScenario] = CONTROL_SCENARIOS

ALL_SCENARIOS: list[ToolScenario] = ALL_TOOL_SCENARIOS + ALL_CONTROL_SCENARIOS


def get_scenarios_by_domain(domain: str) -> list[ToolScenario]:
    """Get all scenarios for a specific tool domain."""
    return [s for s in ALL_SCENARIOS if s.domain == domain]


def get_scenarios_by_ambiguity(ambiguity: str) -> list[ToolScenario]:
    """Get all scenarios for a specific ambiguity level."""
    return [s for s in ALL_SCENARIOS if s.ambiguity == ambiguity]


def get_scenario_stats() -> dict[str, int]:
    """Get counts of scenarios by category."""
    stats = {
        "total": len(ALL_SCENARIOS),
        "tool_scenarios": len(ALL_TOOL_SCENARIOS),
        "control_scenarios": len(ALL_CONTROL_SCENARIOS),
        "by_ambiguity": {},
        "by_domain": {},
    }

    for ambiguity in ["EXPLICIT", "IMPLICIT", "CONTROL"]:
        stats["by_ambiguity"][ambiguity] = len(get_scenarios_by_ambiguity(ambiguity))

    for domain in ["memory_save", "web_search", "code_execute", "file_operation", "api_call", "general"]:
        stats["by_domain"][domain] = len(get_scenarios_by_domain(domain))

    return stats


if __name__ == "__main__":
    # Print scenario stats when run directly
    stats = get_scenario_stats()
    print("Tool Calling Scenario Statistics")
    print("=" * 40)
    print(f"Total scenarios: {stats['total']}")
    print(f"  Tool scenarios: {stats['tool_scenarios']}")
    print(f"  Control scenarios: {stats['control_scenarios']}")
    print()
    print("By ambiguity level:")
    for level, count in stats["by_ambiguity"].items():
        print(f"  {level}: {count}")
    print()
    print("By domain:")
    for domain, count in stats["by_domain"].items():
        print(f"  {domain}: {count}")
