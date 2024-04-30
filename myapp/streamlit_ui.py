import streamlit as st
from myapp import conversation
from streamlit_chat import message

def display_chat_history(chain):
    """ Display chat history using Streamlit components.
    Args:
        chain: Conversational chain.
    """
    # Profile switch section
    st.sidebar.title("Profile Switch")
    profile = st.sidebar.radio(
        "Select Profile:", ("Employee", "Recruiter", "HR", "Manager", "Admin")
    )

    # Profile-specific actions
    actions = {
        "Employee": [
            "Check company policies",
            "Manage attendance",
            "Handle assignments",
            "Track projects",
            "Access salary details",
            "Enroll in benefits",
            "Request time off",
            "Update personal information",
            "Access employee directory",
            "Submit expense reports",
            "Participate in employee surveys",
            "Access training materials",
        ],
        "Recruiter": [
            "Post job openings",
            "Screen applicants",
            "Schedule interviews",
            "Manage candidate pipelines",
            "Conduct pre-employment assessments",
            "Extend job offers",
            "Negotiate salaries and benefits",
            "Onboard new hires",
            "Track recruitment metrics",
            "Manage job boards and advertising",
        ],
        "HR": [
            "Handle employee inquiries",
            "Process leave requests",
            "Conduct performance reviews",
            "Manage employee training programs",
            "Administer compensation and benefits",
            "Ensure compliance with labor laws",
            "Manage employee grievances and disciplinary actions",
            "Coordinate employee engagement activities",
            "Maintain employee records and data",
            "Develop HR policies and procedures",
        ],
        "Manager": [
            "Assign tasks",
            "Track project progress",
            "Manage team schedules",
            "Conduct performance evaluations",
            "Facilitate team communication",
            "Provide coaching and mentoring",
            "Identify training needs for team members",
            "Manage team budgets and resources",
            "Resolve conflicts within the team",
            "Collaborate with cross-functional teams",
        ],
        "Admin": [
            "Access all features for comprehensive management",
            "Manage user accounts",
            "Generate data analytics reports",
            "Configure system settings",
            "Manage company-wide policies and procedures",
            "Oversee IT infrastructure and security",
            "Manage legal and regulatory compliance",
            "Develop strategic plans and objectives",
            "Manage corporate communications",
            "Oversee financial operations and budgeting",
        ],
    }

    st.sidebar.subheader("Actions")
    if profile in actions:
        for action in actions[profile]:
            if st.sidebar.button(action):
                st.session_state["input"] = action

    # Chat history container
    reply_container = st.container()
    container = st.container()

    # Chat history display
    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Query:",
                placeholder="Ask me anything about OnFinance",
                key="input",
                value=st.session_state.get("input", ""),
            )
            submit_button = st.form_submit_button(label="Resolve")
            if submit_button and user_input:
                with st.spinner("Let me check ......"):
                    output = conversation.conversation_chat(
                        query=user_input,
                        chain=chain,
                        history=st.session_state["history"],
                    )
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)

            # Display chat history
            if st.session_state["generated"]:
                with reply_container:
                    for i in range(len(st.session_state["generated"])):
                        user_message = st.session_state["past"][i]
                        chatbot_message = st.session_state["generated"][i]
                        save_button_key = f"save_{i}"
                        if st.session_state.get(save_button_key, False):
                            save_button = st.button(
                                f"Message {i + 1} Saved",
                                key=save_button_key,
                                disabled=True,
                            )
                        else:
                            save_button = st.button(
                                f"Save Message {i + 1}", key=save_button_key
                            )
                        if save_button:
                            conversation.save_message(user_message, chatbot_message)
                            st.session_state[save_button_key] = True

                        message(
                            user_message,
                            is_user=True,
                            key=str(i) + "_user",
                            avatar_style="initials",
                            seed="ME",
                        )
                        message(
                            chatbot_message,
                            key=str(i),
                            avatar_style="initials",
                            seed="OF",
                        )