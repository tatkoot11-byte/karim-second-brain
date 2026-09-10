import chainlit as cl

from agent.graph import run_agent
from agent.memory import (
    add_memory,
    search_memory,
    forget_all,
    classify_memory,
)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "🧠 **Karim's Second-Brain Agent is ready!**\n\n"
            "I can use tools, retrieve memories, and store memories "
            "only after explicit consent."
        )
    ).send()


async def show_step(name: str, output: str):
    step = cl.Step(name=name, type="tool")
    step.output = output
    await step.send()


@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()

    # Handle /forget command
    if text.lower() == "/forget":
        result = forget_all()

        await cl.Message(
            content=result["message"]
        ).send()

        return

    # Retrieve relevant memories
    memory_result = search_memory(text)

    memory_items = []

    if isinstance(memory_result, dict):
        memory_items = memory_result.get("results", [])

    if memory_items:
        memory_summary = "\n".join(
            f"- {item.get('memory', '')}"
            for item in memory_items[:3]
            if item.get("memory")
        )

        await show_step(
            "Memory Retrieval",
            memory_summary,
        )
    else:
        await show_step(
            "Memory Retrieval",
            "No relevant memories found.",
        )

    # Run LangGraph agent
    result = run_agent(text)

    # Show tool activity in Chainlit
    for item in result.get("intermediate_results", []):
        if item.startswith("Tool call:"):
            await show_step(
                "Tool Call",
                item,
            )

        elif item.startswith("Recovery:"):
            await show_step(
                "Recovery",
                item,
            )

    # Final response
    messages = result.get("messages", [])

    if messages:
        final_message = messages[-1].content

        if isinstance(final_message, list):
            final_message = str(final_message)

        await cl.Message(
            content=final_message
        ).send()

    else:
        await cl.Message(
            content="I could not generate a response."
        ).send()

    # Ask for explicit consent before storing memory
    response = await cl.AskActionMessage(
        content="Would you like me to store this request in Karim's memory?",
        actions=[
            cl.Action(
                name="consent_yes",
                payload={"value": "yes"},
                label="✅ Yes, store it",
            ),
            cl.Action(
                name="consent_no",
                payload={"value": "no"},
                label="❌ No, don't store it",
            ),
        ],
    ).send()

    # Process the selected action
    if response:
        action = response.get("name")

        if action == "consent_yes":
            memory_saved = add_memory(
                content=text,
                category=classify_memory(text),
                consent_given=True,
            )

            if memory_saved.get("stored"):
                await cl.Message(
                    content="✅ Memory stored with your explicit consent."
                ).send()
            else:
                await cl.Message(
                    content="⚠️ Memory was not stored."
                ).send()

        else:
            await cl.Message(
                content="🔒 Nothing was stored because you did not give consent."
            ).send()

    else:
        await cl.Message(
            content="🔒 Nothing was stored because you did not give consent."
        ).send()