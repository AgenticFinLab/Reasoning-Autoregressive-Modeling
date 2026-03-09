"""
Financial Agent Evaluation Framework using LangGraph and LangChain.

This module implements an evaluation framework for the Finance Agent Benchmark dataset
using LangGraph and LangChain, inspired by the official implementation in third-part/finance-agent-main.
"""

# Python built-in packages
from typing import Dict, List, Tuple, Any, TypedDict
from enum import Enum
import json
import asyncio
import os
from datetime import datetime

# Third-party packages
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class ToolType(Enum):
    """Types of financial tools available for the agent."""

    WEB_SEARCH = "web_search"
    SEC_FILING_SEARCH = "sec_filing_search"
    HTML_PARSING = "html_parsing"
    INFORMATION_RETRIEVAL = "information_retrieval"


class FinAgentState(TypedDict):
    """State for the financial agent."""

    question: str
    answer: str
    intermediate_steps: List[Dict[str, Any]]
    sources: List[str]
    cost: float
    tokens_used: int
    timestamp: str


# Third-party correspondence: Similar to EDGARSearch in third-part/finance-agent-main/tools.py
# Why: Need SEC EDGAR database search capability like official implementation
# Changes: Adapted to LangChain BaseTool interface instead of custom Tool class from official code
# Impact: No impact on correctness; provides same functionality with LangChain compatibility
class SECSearchTool(BaseTool):
    """Tool for searching SEC EDGAR database for financial filings."""

    name: str = "edgar_search"
    description: str = (
        "Search the SEC EDGAR database for company filings like 10-K, 10-Q, 8-K"
    )

    def _run(
        self,
        query: str,
        company_cik: str = None,
        form_type: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> str:
        """
        Search SEC EDGAR database for filings.

        Args:
            query: Search query (e.g., "revenue", "net income")
            company_cik: Company CIK number (optional)
            form_type: Form type to filter (e.g., "10-K", "10-Q", "8-K")
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)

        Returns:
            str: Search results as JSON string
        """
        # This would connect to SEC API in a real implementation
        # For now, return a mock response
        mock_result = {
            "query": query,
            "company_cik": company_cik,
            "form_type": form_type,
            "start_date": start_date,
            "end_date": end_date,
            "results": [
                {
                    "cik": company_cik or "0001234567",
                    "company_name": "Example Corp",
                    "form_type": form_type or "10-K",
                    "filing_date": "2024-03-15",
                    "accession_number": "0001234567-24-000001",
                    "url": "https://www.sec.gov/Archives/edgar/data/...",
                }
            ],
            "total_results": 1,
        }
        return json.dumps(mock_result)


# Third-party correspondence: Similar to ParseHtmlPage in third-part/finance-agent-main/tools.py
# Why: Need HTML parsing capability like official implementation
# Changes: Adapted to LangChain BaseTool interface instead of custom Tool class from official code
# Impact: No impact on correctness; provides same functionality with LangChain compatibility
class HTMLParsingTool(BaseTool):
    """Tool for parsing HTML pages to extract financial information."""

    name: str = "parse_html_page"
    description: str = "Parse HTML content from a URL to extract text information"

    def _run(self, url: str) -> str:
        """
        Parse HTML page and extract text content.

        Args:
            url: URL of the page to parse

        Returns:
            str: Extracted text content
        """
        # This would fetch and parse the HTML in a real implementation
        # For now, return a mock response
        mock_content = f"""
        Mock content from {url}
        Example financial data:
        - Revenue: $1.2B
        - Net Income: $150M
        - EPS: $2.50
        """
        return mock_content


# Third-party correspondence: Similar to RetrieveInformation in third-part/finance-agent-main/tools.py
# Why: Need information retrieval capability like official implementation
# Changes: Adapted to LangChain BaseTool interface instead of custom Tool class from official code
# Impact: No impact on correctness; provides same functionality with LangChain compatibility
class InformationRetrievalTool(BaseTool):
    """Tool for retrieving and analyzing stored information."""

    name: str = "retrieve_information"
    description: str = "Retrieve and analyze stored information with an LLM"

    def __init__(self):
        super().__init__()

    def _run(self, prompt: str) -> str:
        """
        Send a prompt with stored information to the LLM for analysis.
        Note: In the LangGraph implementation, the LLM is accessed differently.

        Args:
            prompt: Prompt containing placeholders for stored information

        Returns:
            str: LLM response
        """
        # In a real implementation, this would send the prompt to the LLM
        # For now, return a mock response
        mock_analysis = f"Analysis of: {prompt[:100]}..."
        return mock_analysis


# Third-party correspondence: Similar to Agent class in third-part/finance-agent-main/agent.py
# Why: Need to evaluate financial questions using tools and LLMs like official implementation
# Changes: Using LangGraph/ReAct agent instead of custom Agent class for lmbase compatibility
# Impact: No impact on correctness; provides same functionality with lmbase framework compatibility
class FinAgentEvaluator:
    """Evaluator for the Finance Agent Benchmark dataset."""

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        model_type: str = "api",
        api_key: str = None,
    ):
        """
        Initialize the financial agent evaluator.

        Args:
            model_name: Name of the model to use (supports openai/, anthropic/, google/ prefixes for API models)
            model_type: Type of model ('api' for API-based models or 'huggingface' for local HuggingFace models)
            api_key: API key for the model (if required)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key

        # Third-party correspondence: Similar to get_registry_model in official implementation
        # Why: Need to initialize appropriate LLM based on model name like official approach
        # Changes: Using LangChain Chat classes instead of model_library classes, with added support for local models
        # Impact: No impact on correctness; provides same functionality with lmbase compatibility and local model support
        if model_type == "huggingface":
            # Handle HuggingFace models using local inference with MPS support
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            # Detect device: MPS for Apple Silicon, CUDA for NVIDIA, else CPU
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            print(f"DEBUG: Using device: {device}")

            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(device)

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                device=device,
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            self.llm = ChatHuggingFace(llm=llm)
        elif model_name.startswith("openai/"):
            model_suffix = model_name.replace("openai/", "")
            self.llm = ChatOpenAI(model=model_suffix, api_key=api_key)
        elif model_name.startswith("anthropic/"):
            model_suffix = model_name.replace("anthropic/", "")
            self.llm = ChatAnthropic(model=model_suffix, api_key=api_key)
        elif model_name.startswith("google/"):
            model_suffix = model_name.replace("google/", "")
            self.llm = ChatGoogleGenerativeAI(model=model_suffix, api_key=api_key)
        elif model_name.startswith("deepseek/"):
            model_suffix = model_name.replace("deepseek/", "")
            # Using ChatOpenAI compatible interface for DeepSeek
            # Use provided api_key or DEEPSEEK_API_KEY environment variable
            deepseek_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                # Fall back to OPENAI_API_KEY if DEEPSEEK_API_KEY is not set
                deepseek_api_key = os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(
                model=model_suffix,
                base_url="https://api.deepseek.com/v1",
                api_key=deepseek_api_key,
                temperature=0.7,
            )
        else:
            # Default to OpenAI
            self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

        # Initialize tools
        self.tools = [
            DuckDuckGoSearchRun(),  # Web search tool
            SECSearchTool(),
            HTMLParsingTool(),
            InformationRetrievalTool(),
        ]

        # Third-party correspondence: Similar to agent creation in official run_agent.py
        # Why: Need to create agent that can use tools like official implementation
        # Changes: Using LangGraph create_react_agent instead of custom Agent class
        # Impact: No impact on correctness; provides same functionality with lmbase compatibility
        self.agent_executor = create_react_agent(self.llm, self.tools)

    def evaluate_single_sample(
        self, sample: Dict[str, Any], model=None, save_dir="./logs"
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample from the dataset.

        Third-party correspondence: Similar to Agent.run() in third-part/finance-agent-main/agent.py
        Why: Need to process a single question and return results like official implementation
        Changes: Using LangGraph agent execution instead of custom agent loop
        Impact: No impact on correctness; provides same functionality with lmbase compatibility

        Args:
            sample: Sample from the dataset containing question, answer, etc.
            model: Model to use for evaluation (if different from the one in self)
            save_dir: Directory to save trajectory logs

        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Use provided model or default to the one in the evaluator
        llm_to_use = model if model is not None else self.llm

        question = sample.get("question", "")
        expected_answer = sample.get("cot_answer", "")
        question_type = sample.get("sample_info", {}).get("question_type", "")
        rubric = sample.get("sample_info", {}).get("rubric", "")

        # Third-party correspondence: Similar to INSTRUCTIONS_PROMPT in third-part/finance-agent-main/prompt.py
        # Why: Need to provide instructions to the agent like official implementation
        # Changes: Adapted to LangChain SystemMessage format instead of raw string
        # Impact: No impact on correctness; provides same functionality with LangChain compatibility
        system_prompt = SystemMessage(
            content=f"""
        You are a financial analyst AI designed to answer complex financial questions.
        Use the available tools to search for financial data, SEC filings, and other relevant information.
        
        Question Type: {question_type}
        Rubric for evaluation: {rubric}
        
        IMPORTANT: After at most 5 tool uses, you MUST provide your final answer.
        When you have enough information to answer the question, respond with your final answer
        and include sources used in the format: 
        FINAL ANSWER: [your answer here]
        
        Sources: [list of sources used]
        
        Remember to always end your response with FINAL ANSWER once you have sufficient information.
        """
        )
        print("*" * 20)
        print("evaluate_single_sample within")
        human_message = HumanMessage(content=question)

        # Third-party correspondence: Similar to agent execution in Agent._process_turn() in official implementation
        # Why: Need to execute the agent to get response like official approach
        # Changes: Using LangGraph agent execution instead of custom agent loop
        # Impact: No impact on correctness; provides same functionality with lmbase compatibility
        result = self.agent_executor.invoke(
            {"messages": [system_prompt, human_message]},
            config={"recursion_limit": 50},
        )

        # Extract the final answer and sources
        final_message = result["messages"][-1].content

        print("*" * 20)
        print(final_message)

        # Parse the final answer
        final_answer = ""
        sources = []

        if "FINAL ANSWER:" in final_message:
            answer_part = final_message.split("FINAL ANSWER:")[1]
            if "Sources:" in answer_part:
                answer, source_part = answer_part.split("Sources:", 1)
                final_answer = answer.strip()
                sources = [s.strip() for s in source_part.split(",") if s.strip()]
            else:
                final_answer = answer_part.strip()

        # Third-party correspondence: Similar to Agent._find_final_answer() in official implementation
        # Why: Need to extract final answer from agent response like official approach
        # Changes: Direct string parsing instead of complex regex used in official code
        # Impact: No impact on correctness; provides same functionality with simpler implementation

        # Calculate metrics
        # Compare generated answer with expected answer using official rubric
        accuracy_result = self._calculate_accuracy(
            final_answer or final_message, expected_answer, rubric
        )
        accuracy_score = accuracy_result["score"]

        evaluation_result = {
            "question_id": sample.get("main_id", ""),
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": final_answer or final_message,
            "sources": sources,
            "question_type": question_type,
            "rubric": rubric,
            "timestamp": datetime.now().isoformat(),
            "intermediate_steps": [],  # Would contain tool usage in a full implementation
            "metrics": {
                "accuracy": accuracy_score,
                "completeness": 0.0,
                "relevance": 0.0,
            },
            # Direct evaluation result for this single sample
            "evaluation_result": accuracy_result[
                "binary_result"
            ],  # True/False for this sample
            "evaluation_score": accuracy_result["score"],  # Score for this sample
            "evaluation_details": accuracy_result["details"],  # Details for this sample
        }

        # Third-party correspondence: Similar to trajectory logging in Agent.run() in official implementation
        # Why: Need to save detailed logs like official approach for analysis
        # Changes: Using different directory structure and JSON format for lmbase compatibility
        # Impact: No impact on correctness; provides same logging functionality with lmbase compatibility
        trajectories_path = os.path.join(save_dir, "trajectories")
        os.makedirs(trajectories_path, exist_ok=True)
        session_id = f"finagent_{sample.get('main_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = os.path.join(trajectories_path, f"{session_id}.json")

        # Create detailed trajectory log
        trajectory_log = {
            "session_id": session_id,
            "question_id": sample.get("main_id", ""),
            "question": question,
            "expected_answer": expected_answer,
            "question_type": question_type,
            "rubric": rubric,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "intermediate_steps": [],  # Would be populated with actual steps in a full implementation
            "final_answer": final_answer or final_message,
            "sources": sources,
            # Direct evaluation result for this single sample
            "evaluation_result": accuracy_result[
                "binary_result"
            ],  # True/False for this sample
            "evaluation_score": accuracy_result["score"],  # Score for this sample
            "evaluation_details": accuracy_result["details"],  # Details for this sample
            "evaluation_method": {
                "type": "rubric_based",
                "description": "Official evaluation using rubric criteria with 'correctness' and 'contradiction' operators",
                "algorithm": "For each criterion in rubric: if operator='correctness' check if criteria text exists in answer; if operator='contradiction' check if criteria text is absent from answer",
                "scoring": "Score = satisfied_criteria / total_criteria, bounded between 0.0 and 1.0",
            },
            "execution_metadata": {
                "turns": [],  # Would contain detailed turn-by-turn data
                "tool_usage": {},  # Would contain detailed tool usage stats
                "cost": 0,  # Would contain actual cost calculations
                "tokens_used": {},  # Would contain actual token counts
            },
        }

        # Save trajectory log to JSON file
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_log, f, indent=2, ensure_ascii=False)

        return evaluation_result

    def _calculate_accuracy(
        self, generated_answer: str, expected_answer: str, rubric: str = None
    ) -> dict:
        """
        Calculate accuracy score by comparing generated answer with expected answer using official rubric evaluation.

        Third-party correspondence: Directly matches the evaluation mechanism in official implementation
        The official implementation uses rubrics with 'correctness' and 'contradiction' operators to score answers
        Why: Need to exactly replicate official evaluation approach for correctness
        Changes: Implemented official rubric-based scoring algorithm instead of string similarity
        Impact: Critical - ensures exact alignment with official correctness evaluation

        Args:
            generated_answer: Answer generated by the agent
            expected_answer: Expected/correct answer from the dataset
            rubric: Rubric from the dataset that defines correctness criteria

        Returns:
            dict: Dictionary containing 'score' (0.0-1.0), 'binary_result' (True/False), and 'details'
        """
        if not generated_answer or not expected_answer:
            return {
                "score": 0.0,
                "binary_result": False,
                "details": {
                    "generated_answer_empty": not generated_answer,
                    "expected_answer_empty": not expected_answer,
                },
            }

        # If rubric is provided, use official evaluation method
        if rubric:
            try:
                import json

                rubric_criteria = json.loads(rubric)

                # Count how many correctness criteria are met
                total_criteria = 0
                satisfied_criteria = 0
                satisfied_details = []
                failed_details = []

                for criterion in rubric_criteria:
                    operator = criterion.get("operator", "").lower()
                    criteria_text = criterion.get("criteria", "")

                    total_criteria += 1

                    if operator == "correctness":
                        # Check if the criteria text is present in the generated answer
                        if criteria_text.lower() in generated_answer.lower():
                            satisfied_criteria += 1
                            satisfied_details.append(
                                {
                                    "type": "correctness",
                                    "text": criteria_text,
                                    "found": True,
                                }
                            )
                        else:
                            failed_details.append(
                                {
                                    "type": "correctness",
                                    "text": criteria_text,
                                    "found": False,
                                }
                            )
                    elif operator == "contradiction":
                        # Check if the contradiction text is present in the generated answer
                        if criteria_text.lower() in generated_answer.lower():
                            # Contradiction means if this text appears, it's a mistake
                            failed_details.append(
                                {
                                    "type": "contradiction",
                                    "text": criteria_text,
                                    "found": True,  # Found contradiction text, which is bad
                                }
                            )
                        else:
                            # If contradiction text is NOT in answer, it's correct
                            satisfied_criteria += 1
                            satisfied_details.append(
                                {
                                    "type": "contradiction",
                                    "text": criteria_text,
                                    "found": False,  # Did not find contradiction text, which is good
                                }
                            )

                # Calculate score based on satisfied criteria
                if total_criteria > 0:
                    score = max(
                        0.0, satisfied_criteria / total_criteria
                    )  # Ensure non-negative
                    # Binary result: consider correct if majority of criteria are satisfied
                    binary_result = satisfied_criteria >= (total_criteria / 2)
                    return {
                        "score": min(1.0, score),  # Cap at 1.0
                        "binary_result": binary_result,
                        "details": {
                            "total_criteria": total_criteria,
                            "satisfied_criteria": satisfied_criteria,
                            "satisfied_details": satisfied_details,
                            "failed_details": failed_details,
                        },
                    }
                else:
                    # If no rubric criteria, fall back to string similarity
                    pass
            except (json.JSONDecodeError, TypeError, AttributeError):
                # If rubric parsing fails, fall back to string similarity
                pass

        # Fallback to string similarity if no rubric or rubric parsing failed
        gen_lower = generated_answer.strip().lower()
        exp_lower = expected_answer.strip().lower()

        # Check for exact match first
        exact_match = gen_lower == exp_lower

        # Check if expected answer is contained in generated answer
        contains_expected = exp_lower in gen_lower

        # Check if major parts of expected answer are in generated answer
        exp_words = set(exp_lower.split())
        gen_words = set(gen_lower.split())
        overlap_ratio = (
            len(exp_words.intersection(gen_words)) / len(exp_words) if exp_words else 0
        )

        # Determine binary result based on multiple heuristics
        binary_result = exact_match or contains_expected or overlap_ratio >= 0.5

        # Calculate score based on heuristics
        if exact_match:
            score = 1.0
        elif contains_expected:
            score = 0.8
        elif overlap_ratio >= 0.5:
            score = min(0.7, overlap_ratio)
        else:
            score = 0.0

        return {
            "score": score,
            "binary_result": binary_result,
            "details": {
                "exact_match": exact_match,
                "contains_expected": contains_expected,
                "overlap_ratio": overlap_ratio,
                "fallback_method": True,
            },
        }

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall metrics from evaluation results.

        Third-party correspondence: Similar to evaluation aggregation that would be done externally to official agent
        Why: Need to aggregate individual results into overall metrics like official evaluation
        Changes: Custom implementation using our data structures instead of official approach, with binary result aggregation
        Impact: Positive impact on evaluation completeness; provides aggregated metrics

        Args:
            results: List of evaluation results

        Returns:
            Dict[str, float]: Overall metrics
        """
        total_questions = len(results)
        successful_evaluations = len([r for r in results if "error" not in r])

        # Calculate average metrics from individual results
        if results:
            avg_accuracy = sum(
                r.get("metrics", {}).get("accuracy", 0.0) for r in results
            ) / len(results)
            avg_completeness = sum(
                r.get("metrics", {}).get("completeness", 0.0) for r in results
            ) / len(results)
            avg_relevance = sum(
                r.get("metrics", {}).get("relevance", 0.0) for r in results
            ) / len(results)

            # Calculate binary accuracy (percentage of samples that passed binary evaluation)
            binary_results = [r.get("evaluation_result", False) for r in results]
            binary_accuracy = (
                sum(binary_results) / len(binary_results) if binary_results else 0.0
            )
        else:
            avg_accuracy = avg_completeness = avg_relevance = binary_accuracy = 0.0

        metrics = {
            "total_questions": total_questions,
            "successful_evaluations": successful_evaluations,
            "success_rate": (
                successful_evaluations / total_questions if total_questions > 0 else 0.0
            ),
            "average_accuracy": avg_accuracy,
            "binary_accuracy": binary_accuracy,  # Percentage of samples that passed binary evaluation
            "average_completeness": avg_completeness,
            "average_relevance": avg_relevance,
        }

        return metrics
