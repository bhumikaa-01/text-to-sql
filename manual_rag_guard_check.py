import asyncio
from agent.sql_chain import run_query

questions = [
    "How much money did the company make?",
    "How many orders were canceled?",
    "Which product generated the most sales?",
    "How many customers do we have?",
    "What was the average order value?",
]

async def main():
    for question in questions:
        result = await run_query(question)

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("SQL:", result["sql"])
        print("ERROR:", result["error"])
        print("GUARD:", result["resource_guard"]["decision"])
        print("CONFIDENCE:", result["confidence"]["score"])

asyncio.run(main())

