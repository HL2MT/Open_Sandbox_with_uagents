import os
import streamlit as st
import google.generativeai as genai
import time
from collections import deque
import json
import git
import requests
from uagents import Agent
import asyncio


# Configure the Gemini model
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "gemini Api Key"  # Replace with your actual API key file path  link: https://aistudio.google.com/app/prompts/new_chat?gad_source=1&gclid=Cj0KCQjwpvK4BhDUARIsADHt9sRAYLFVfLEwMG-Yz5wwJ6p3JozW8DB2hK-eqmP9uYOLJ12Qw8vNheAaAq4pEALw_wcB
genai.configure(api_key=os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
model = genai.GenerativeModel('gemini-1.5-flash') #1M context window but 15 call/min
model_for_workflow_analysation = genai.GenerativeModel('gemini-1.5-pro-exp-0827') #2M contet window but 2 call/min

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)





# Initialize session states
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_call_timestamps' not in st.session_state:
    st.session_state.api_call_timestamps = deque(maxlen=15)
if 'repositories' not in st.session_state:
    st.session_state.repositories = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'show_file_browser' not in st.session_state:
    st.session_state.show_file_browser = False
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'show_functionalities' not in st.session_state:
    st.session_state.show_functionalities = False


#design a custom class with Agent from uagent as super class
class analyzer_agent_(Agent):
    def __init__(self, name):
        super().__init__(name)

    #dummy function
    def printer(self,stm):
        st.write(stm)

    #first check for internet connect for repository cloning
    @staticmethod
    def check_internet_connection():
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except requests.ConnectionError:
            return False

    # New function to clone a repository
    @staticmethod
    def clone_repository(repo_url, local_path):
        try:
            if os.path.exists(os.path.join(local_path, '.git')):
                # Repository already exists, update it
                repo = git.Repo(local_path)
                origin = repo.remotes.origin
                st.info(f"Updating existing repository in {local_path}")
                origin.fetch()
                origin.pull()
                st.success(f"Repository in {local_path} updated successfully")
            else:
                # Repository doesn't exist, clone it
                st.info(f"Cloning repository to {local_path}")
                git.Repo.clone_from(repo_url, local_path)
                st.success(f"Repository cloned successfully to {local_path}")
            return True
        except git.GitCommandError as e:
            error_message = str(e)
            if "Could not resolve host: github.com" in error_message:
                st.error("Unable to connect to GitHub. Please check your internet connection.")
                if not analyzer_agent_.check_internet_connection():
                    st.error("No internet connection detected. Please check your network settings.")
                else:
                    st.error(
                        "Internet connection is available, but GitHub cannot be reached. This might be a temporary issue with GitHub or your DNS settings.")
                st.info("Suggestions:")
                st.info("1. Check your internet connection")
                st.info("2. Try accessing https://github.com in your web browser")
                st.info("3. If the problem persists, try flushing your DNS cache or changing your DNS server")
            else:
                st.error(f"Git operation failed: {error_message}")
            return False
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return False


    # Rate limiter function because free version only allow 15 call/min
    @staticmethod
    def can_make_api_call():
        current_time = time.time()
        if len(st.session_state.api_call_timestamps) < 15:
            st.session_state.api_call_timestamps.append(current_time)
            return True

        oldest_call = st.session_state.api_call_timestamps[0]
        if current_time - oldest_call >= 60:  # 60 seconds = 1 minute
            st.session_state.api_call_timestamps.append(current_time)
            return True
        return False

    @staticmethod
    def get_file_structure(directory_path):
        """Generate a structured representation of the directory."""
        structure = []
        for root, dirs, files in os.walk(directory_path):
            level = root.replace(directory_path, '').count(os.sep)
            indent = '  ' * level
            structure.append(f"{indent}{os.path.basename(root)}/")
            for file in files:
                structure.append(f"{indent}  {file}")
        return '\n'.join(structure)

    @staticmethod
    def get_file_content(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            return "[Binary file]"

    @staticmethod
    def get_file_content_summary(file_path):
        """Read and return a summary of the file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return f"File content summary:\n{content[:2000]}..." if len(content) > 2000 else content
        except UnicodeDecodeError:
            return "[Binary file]"

    @staticmethod
    def analyze_directory_workflow(directory_path):
        """Generate a workflow analysis for the directory, using cache if available."""
        if directory_path in st.session_state.analysis_cache:
            return st.session_state.analysis_cache[directory_path]

        structure = analyzer_agent_.get_file_structure(directory_path)
        file_summaries = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory_path)
                summary = analyzer_agent_.get_file_content_summary(file_path)
                file_summaries.append(f"File: {relative_path}\n{summary}\n")

        prompt = f"""Analyze the following directory structure and provide a detailed workflow analysis:

        Directory Structure:
        {structure}

        File Summaries:
        {''.join(file_summaries)}

        Please provide:
        1. Give a detailed overview of the directory, what it does?, overview of all its sub-directories and files.
        2. A comprehensive description of the directory's workflow, including how data or control flows between different components.
        """

        try:
            response = model.generate_content(prompt)
            if response.text:
                analysis = response.text
                st.session_state.analysis_cache[directory_path] = analysis
                return analysis
            else:
                return "Workflow analysis blocked due to content safety measures. Please check the directory content."
        except Exception as e:
            return f"Error during workflow analysis: {str(e)}"

    def analyze_file_workflow(file_path):
        """Analyze file workflow, using cache if available."""
        if file_path in st.session_state.analysis_cache:
            return st.session_state.analysis_cache[file_path]

        file_content = analyzer_agent_.get_file_content(file_path)

        prompt = f"""Analyze the following code file and provide a detailed workflow analysis:

        file_content:
        {file_content}
        Please provide:
        1. Give a detailed overview of the file, what it does?, overview of complete file and its dependencies and detailed workflow with condition.
        2. Detailed explanation of each and every class, functions and their dependencies, flow with other class and functions.
        """

        try:
            response = model.generate_content(prompt)
            if response.text:
                analysis = response.text
                st.session_state.analysis_cache[file_path] = analysis
                return analysis
            else:
                return "Workflow analysis blocked due to content safety measures. Please check the file content."
        except Exception as e:
            return f"Error during workflow analysis: {str(e)}"

    @staticmethod
    def list_files_and_dirs(directory):
        """List directories first and then files in the given directory."""
        try:
            items = os.listdir(directory)
            dirs = [item for item in items if os.path.isdir(os.path.join(directory, item))]
            files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
            return dirs, files
        except Exception as e:
            st.error(f"Error: {e}")
            return [], []

    @staticmethod
    def display_directory_contents(directory):
        """Display the contents of the directory."""
        dirs, files = analyzer_agent_.list_files_and_dirs(directory)

        for dir_name in dirs:
            dir_path = os.path.join(directory, dir_name)
            if st.sidebar.button(f"📁 {dir_name}", key=dir_path):
                st.session_state.current_path = dir_path
                st.session_state.selected_file = None

        for file_name in files:
            file_path = os.path.join(directory, file_name)
            if st.sidebar.button(f"📄 {file_name}", key=file_path):
                st.session_state.selected_file = file_path

        analyze_checkbox = st.sidebar.checkbox("Analyze this directory")
        if analyze_checkbox:
            st.write(analyzer_agent_.analyze_directory_workflow(directory))


    @staticmethod
    def display_file_contents(file_path):
        """Display the contents of the selected file."""
        c1, c2 = st.columns([0.55, 0.45])
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            c1.header(f"{os.path.basename(file_path)}")
            with c1.container(height=1000):
                st.code(content)
            analyze_checkbox = st.sidebar.checkbox("Analyze file workflow")
            if analyze_checkbox:
                c2.header("Analysis of File")
                with c2.container(height=1000):
                    st.write(analyzer_agent_.analyze_file_workflow(file_path))
        except Exception as e:
            st.error(f"Error reading file {file_path}: {e}")

    #function to handle chat queries
    @staticmethod
    def process_chat_query(query, root_directory):
        if not analyzer_agent_.can_make_api_call():
            return "Rate limit reached. Please wait a moment before sending another query."

        # Prepare context from repository
        structure = analyzer_agent_.get_file_structure(root_directory)
        relevant_files = []

        # Search for relevant files based on query
        for root, _, files in os.walk(root_directory):
            for file in files:
                file_path = os.path.join(root, file)
                content = analyzer_agent_.get_file_content(file_path)
                if any(keyword in content.lower() for keyword in query.lower().split()):
                    relative_path = os.path.relpath(file_path, root_directory)
                    relevant_files.append((relative_path, content))

        # Prepare prompt with context
        prompt = f"""Based on the following repository context, please answer this question: {query}

        Repository structure:
        {structure}

        Relevant files found:
        """

        for file_path, content in relevant_files[:3]:  # Limit to 3 most relevant files
            prompt += f"\nFile: {file_path}\n{content[:500]}...\n"

        try:
            response = model.generate_content(prompt)
            return response.text if response.text else "Unable to generate a response."
        except Exception as e:
            return f"Error processing query: {str(e)}"


    @staticmethod
    def process_single_repo_query(query, repo_analyzer):
        """Process a query for a single repository."""
        relevant_components = repo_analyzer.find_relevant_components(query)
        detailed_analyses = {}

        for line in relevant_components.split('\n'):
            for component in os.listdir(repo_analyzer.root_directory):
                component_path = os.path.join(repo_analyzer.root_directory, component)
                rel_path = os.path.relpath(component_path, repo_analyzer.root_directory)
                if component.lower() in line.lower():
                    analysis = repo_analyzer.analyze_component(rel_path)
                    detailed_analyses[rel_path] = analysis

        answer_prompt = f"""Based on this analysis of the repository:

            Relevant Components:
            {relevant_components}

            Detailed Analyses:
            {json.dumps(detailed_analyses, indent=2)}

            Please answer this question about the repository: {query}

            Include in your answer:
            1. Direct response to the question
            2. References to specific parts of the repository
            3. Any important relationships between components
            """

        if analyzer_agent_.can_make_api_call():
            response = model.generate_content(answer_prompt)
            return {
                'answer': response.text,
                'analysis': {
                    'relevant_components': relevant_components,
                    'detailed_analyses': detailed_analyses
                }
            }
        else:
            return {
                'answer': "Rate limit reached. Please wait a moment.",
                'analysis': {
                    'relevant_components': relevant_components,
                    'detailed_analyses': detailed_analyses
                }
            }
    @staticmethod
    def process_query(query, analyzers):
        """Process a query using multiple repository analyzers."""
        if not analyzers:
            return "No repositories analyzed yet. Please add and analyze repositories first."

        all_relevant_components = {}
        all_detailed_analyses = {}

        for repo_name, analyzer in analyzers.items():
            if not analyzer.repo_overview:
                continue

            # 1. Find relevant components
            relevant_components = analyzer.find_relevant_components(query)
            all_relevant_components[repo_name] = relevant_components

            # 2. Analyze relevant components in detail
            detailed_analyses = {}
            for line in relevant_components.split('\n'):
                for component in os.listdir(analyzer.root_directory):
                    component_path = os.path.join(analyzer.root_directory, component)
                    rel_path = os.path.relpath(component_path, analyzer.root_directory)
                    if component.lower() in line.lower():
                        analysis = analyzer.analyze_component(rel_path)
                        detailed_analyses[rel_path] = analysis
            all_detailed_analyses[repo_name] = detailed_analyses

        # 3. Generate answer using detailed analyses from all repositories
        answer_prompt = f"""Based on these analyses across multiple repositories:

        {', '.join(analyzers.keys())}

        Relevant Components:
        {json.dumps(all_relevant_components, indent=2)}

        Detailed Analyses:
        {json.dumps(all_detailed_analyses, indent=2)}

        Please answer this question: {query}

        Include in your answer:
        1. Direct response to the question
        2. References to specific parts of the repositories
        3. Any important relationships between components or repositories
        """

        if analyzer_agent.can_make_api_call():
            response = model.generate_content(answer_prompt)
            return {
                'answer': response.text,
                'relevant_components': all_relevant_components,
                'detailed_analyses': all_detailed_analyses
            }
        else:
            return {
                'answer': "Rate limit reached. Please wait a moment.",
                'relevant_components': all_relevant_components,
                'detailed_analyses': all_detailed_analyses
            }
    @staticmethod
    def summarizer(history):
        """This function is useful to minimize the history so that history don't get too long to fit in context window """
        prompt = f"""This is the chat history about repository give a detailed summarized history 
        without losing any relevant and important information 
        {history}"""
        try:
            response = model.generate_content(prompt)
            if response.text:
                analysis = response.text
                return analysis
            else:
                return "Workflow analysis blocked due to content safety measures. Please check the directory content."
        except Exception as e:
            return f"Error during workflow analysis: {str(e)}"

    @staticmethod
    def save_to_json(data, filename):
        os.makedirs("output", exist_ok=True)
        with open(f"output/{filename}", "w") as f:
            json.dump(data, f, indent=4)
        st.success(f"Data saved to output/{filename}")

    @staticmethod
    def analyze_repository_workflow(repo_path):
        """Generate a workflow analysis for the repository."""
        structure = analyzer_agent_.get_file_structure(repo_path)

        file_summaries = []

        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                summary = analyzer_agent_.get_file_content_summary(file_path)
                file_summaries.append(f"File: {relative_path}\n{summary}\n")

        prompt = f"""Analyze the following GitHub repository structure and provide a detailed functional analysis:
        Repository Structure:
        {structure}

        File Summaries:
        {''.join(file_summaries)}

        Please provide:
        1. A detailed description of the repository's workflow, including how the whole repo works from one end to final end and how data or control flows between different components.
        2. A complete and detailed workflow analysis with all the conditions included in repo.
        3. Every process or operation that occurs within the system.
        4. Any important decision points or conditional flows in the system.
        5. Each and every functionality present in the repo. For important functionalities use high level and for common functionality use low level. Don't miss even a single functionality in repo, give as much as you can.

        For each functionality, use the  json with following format:
        FUNCTIONALITY: [Name of functionality]
        DESCRIPTION: [Description of functionality]
        LEVEL: [High level/Low level]
        USAGE: [How and where it is used]
        BASE_DIRECTORY: [Base directory of the functionality]
        USED_IN: [file1.py, file2.py]
        DEFINED_IN: [file3.py, file4.py]

        Don't use a single word after it

        """

        try:
            response = model.generate_content(prompt)
            if response.text:
                return response.text
            else:
                return "Analysis blocked due to content safety measures. Please check the repository content."
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    @staticmethod
    def functionality_extractor(repository_analysis):
        fucn = {}

        for i in range(1, len(repository_analysis.split("FUNCTIONALITY:"))):
            fucn[repository_analysis.split("FUNCTIONALITY:")[i].split("\n")[0]] = \
            repository_analysis.split("FUNCTIONALITY:")[i].split("\n")[1:]
        # print(analyze_repository.split("FUNCTIONALITY:")[i].split("\n")[:])

        import re
        import json
        int_fucn = {}
        for key, value in fucn.items():
            new_dictionary = {}
            for val in value:
                if "DESCRIPTION" in val:
                    new_val = val.split(":")[:]
                    new_dictionary[re.sub(r'[^a-zA-Z0-9,/_\-\.]', '', new_val[0])] = new_val[1:]
                elif "BASE_DIRECTORY" in val:
                    bs_val = val.split(":")[:]
                    new_vl = []
                    for itm in bs_val[1:]:
                        new_vl.append(re.sub(r'[^a-zA-Z0-9,/_\-\.]', '', itm))
                    print(new_vl)

                    new_dictionary[re.sub(r'[^a-zA-Z0-9,/_\-\.]', '', bs_val[0])] = new_vl
            int_fucn[re.sub(r'[^a-zA-Z0-9,/_\-\.]', '', key)] = new_dictionary
            #return int_fucn

            output_file='output.json'
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(int_fucn, outfile)
        print(int_fucn)

    class RepositoryAnalyzer:
        def __init__(self, root_directory):
            self.root_directory = root_directory
            self.repo_overview = None
            self.analyzed_components = {}

        def build_repository_overview(self):
            """Build a comprehensive overview of the repository structure."""
            structure = analyzer_agent_.get_file_structure(self.root_directory)

            overview_prompt = f"""Analyze this repository structure and create a detailed overview:
            {structure}

            For each directory, provide:
            1. The likely purpose and role in the project
            2. Key files it might contain
            3. Potential relationships with other directories

            Also include:
            1. Overall repository purpose
            2. Main components and their roles
            3. Likely workflows or processes

            Make this overview detailed enough to help locate relevant information when answering questions.
            """

            if analyzer_agent_.can_make_api_call():
                response = model.generate_content(overview_prompt)
                self.repo_overview = response.text

        def find_relevant_components(self, query):
            """Use the repository overview to identify relevant directories or files."""
            if not analyzer_agent_.can_make_api_call():
                return None, "Rate limit reached. Please wait a moment."

            location_prompt = f"""Given this repository overview:
            {self.repo_overview}

            And this query: '{query}'

            Identify the most relevant directories or files to answer this query.
            Explain why each component is relevant.

            Format:
            1. List each relevant directory or file
            2. Explain why it's relevant to the query
            3. Suggest what specific information we might find there
            """

            response = model.generate_content(location_prompt)
            print(response.text)
            return response.text

        def analyze_component(self, component_path):
            """Perform detailed analysis of a specific directory or file."""
            if component_path in self.analyzed_components:
                return self.analyzed_components[component_path]

            if not analyzer_agent_.can_make_api_call():
                return "Rate limit reached. Please wait a moment."

            full_path = os.path.join(self.root_directory, component_path)

            if os.path.isdir(full_path):
                return self._analyze_directory(full_path, component_path)
            elif os.path.isfile(full_path):
                return self._analyze_file(full_path, component_path)
            else:
                return f"Component {component_path} not found."

        def _analyze_directory(self, dir_path, relative_path):
            """Analyze a specific directory in detail."""
            structure = analyzer_agent_.get_file_structure(dir_path)
            files_content = ""

            # Get content of important files in the directory
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if self._is_important_file(file):
                        file_path = os.path.join(root, file)
                        content = analyzer_agent_.get_file_content(file_path)
                        files_content += f"\nFile: {file}\n{content[:1000]}"  # Limit content for API

            analysis_prompt = f"""Analyze this directory in detail:
            Directory: {relative_path}

            Structure:
            {structure}

            Key Files Content:
            {files_content}

            Provide:
            1. Detailed purpose and functionality of this directory
            2. Key components and their roles
            3. How this directory interacts with other parts of the repository
            4. Main workflows or processes within this directory
            """

            response = model.generate_content(analysis_prompt)
            analysis = response.text
            self.analyzed_components[relative_path] = analysis
            return analysis

        def _analyze_file(self, file_path, relative_path):
            """Analyze a specific file in detail."""
            content = analyzer_agent_.get_file_content(file_path)

            analysis_prompt = f"""Analyze this file in detail:
            File: {relative_path}

            Content:
            {content[:2000]}  # Limit content for API

            Provide:
            1. Detailed purpose and functionality of this file
            2. Key components (classes, functions, etc.) and their roles
            3. How this file interacts with other parts of the repository
            4. Main workflows or processes this file is involved in
            """

            response = model.generate_content(analysis_prompt)
            analysis = response.text
            self.analyzed_components[relative_path] = analysis
            return analysis

        def _is_important_file(self, filename):
            """Determine if a file is important enough to analyze in detail."""
            important_extensions = ['.py', '.js', '.java', '.cpp', '.h', '.cs', '.go']
            return any(filename.endswith(ext) for ext in important_extensions)
    @staticmethod
    def json_reader():
        output_file = 'output.json'
        with open('output.json', 'r') as outfile:
            file_content = outfile.read()
            if file_content:
                data = json.loads(file_content)
                # Add your processing logic here
                for key, value in data.items():

                    if 'DESCRIPTION' in value:
                        print(value['DESCRIPTION'])
                    if "BASE_DIRECTORY" in value:
                        st.sidebar.write(key, ":", value["BASE_DIRECTORY"])


analyzer_agent=analyzer_agent_(name='Just for analyzer agent for Open SandBox')









def main():
    st.sidebar.title("Multi-Repository Analyzer")

    # Repository management
    st.sidebar.header("Manage Repositories")
    repo_url = st.sidebar.text_input("Enter Repository URL")
    repo_name = st.sidebar.text_input("Enter Repository Name")
    if st.sidebar.button("Add Repository"):
        if repo_url and repo_name:
            local_path = os.path.join("repositories", repo_name)
            if analyzer_agent.clone_repository(repo_url,local_path):
                st.session_state.repositories[repo_name] = analyzer_agent.RepositoryAnalyzer(local_path)
                with st.spinner(f"Building overview for {repo_name}..."):
                    st.session_state.repositories[repo_name].build_repository_overview()
                st.success(f"Repository {repo_name} added and analyzed.")
            else:
                st.error("Failed to add repository.")
        else:
            st.warning("Please enter both repository URL and name.")

    # Repository selection
    selected_repo = st.sidebar.selectbox("Select Repository", list(st.session_state.repositories.keys()),
                                         key="selected_repo")

    # Dynamic sidebar content based on selected repository
    if selected_repo:
        st.sidebar.header(f"{selected_repo} Options")

        # File browser options
        if st.sidebar.checkbox("Show File Browser"):
            st.session_state.show_file_browser = True
            root_directory = st.session_state.repositories[selected_repo].root_directory
            if 'current_path' not in st.session_state:
                st.session_state.current_path = root_directory
            if 'selected_file' not in st.session_state:
                st.session_state.selected_file = None

            st.sidebar.write(f"Current Path: {st.session_state.current_path}")
            if st.session_state.current_path != root_directory:
                if st.sidebar.button("🔙 Go Back"):
                    parent_path = os.path.dirname(st.session_state.current_path)
                    st.session_state.current_path = parent_path
                    st.session_state.selected_file = None

        # Analysis options
        if st.sidebar.checkbox("Analyze Repository"):
            st.session_state.show_analysis = True
            if st.sidebar.button("Run Workflow Analysis"):
                with st.spinner(f"Analyzing repository {selected_repo}..."):
                    root_directory = st.session_state.repositories[selected_repo].root_directory
                    st.session_state.analysis_results[selected_repo] = analyzer_agent.analyze_repository_workflow(root_directory)
                st.success("Analysis complete!")

        if st.sidebar.checkbox("Show Functionalities"):
            st.session_state.show_functionalities = True

        # Repository info
        st.sidebar.header("Repository Info")
        st.sidebar.write(f"Location: {st.session_state.repositories[selected_repo].root_directory}")
        # Add more repository info here as needed
    # Create three tabs: File Browser, Chat, and Workflow and Features
    tab1, tab2, tab3= st.tabs(["File Browser", "Chat", "Workflow and Features"])

    with tab1:
        if selected_repo and st.session_state.get('show_file_browser', False):
            analyzer_agent.display_directory_contents(st.session_state.current_path)
            if st.session_state.selected_file:
                analyzer_agent.display_file_contents(st.session_state.selected_file)
        else:
            st.info("Please select a repository and enable the file browser in the sidebar.")

    with tab2:
        st.header("Chat with Your Repositories")

        # Chat mode selection
        chat_mode = st.radio("Select Chat Mode", ["General Chat", "Repository-Specific Chat","Code Generator"])

        if chat_mode == "General Chat":
            # Existing general chat logic
            if st.session_state.repositories:
                with st.expander("Repositories Overview"):
                    for repo_name, repo_analyzer in st.session_state.repositories.items():
                        st.subheader(repo_name)
                        st.write(repo_analyzer.repo_overview)

            # Display general chat history
            for message in st.session_state.chat_history[-10:]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "relevant_components" in message:
                        with st.expander("Analysis Process"):
                            st.subheader("1. Relevant Components Identified")
                            st.write(message["relevant_components"])
                            st.subheader("2. Detailed Analyses")
                            for repo, analyses in message["detailed_analyses"].items():
                                st.write(f"**Repository: {repo}**")
                                for component, analysis in analyses.items():
                                    st.write(f"Component: {component}")
                                    st.write(analysis)

            # General chat input
            if prompt := st.chat_input("Ask about all your repositories"):
                with st.chat_message("user"):
                    st.write(prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    response = analyzer_agent.process_query(prompt, st.session_state.repositories)
                    st.write(response['answer'])
                    with st.expander("Analysis Process"):
                        st.subheader("1. Relevant Components Identified")
                        st.write(response["relevant_components"])
                        st.subheader("2. Detailed Analyses")
                        for repo, analyses in response["detailed_analyses"].items():
                            st.write(f"**Repository: {repo}**")
                            for component, analysis in analyses.items():
                                st.write(f"Component: {component}")
                                st.write(analysis)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "relevant_components": response["relevant_components"],
                    "detailed_analyses": response["detailed_analyses"]
                })

                # Keep only the most recent 5 interactions (10 messages) in the chat history
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]

        elif chat_mode == "Repository-Specific Chat": # Repository-Specific Chat

            selected_repo = st.selectbox("Select Repository for Chat", list(st.session_state.repositories.keys()))

            if selected_repo:
                if selected_repo not in st.session_state.repo_chat_histories:
                    st.session_state.repo_chat_histories[selected_repo] = []

                with st.expander(f"{selected_repo} Overview"):
                    st.write(st.session_state.repositories[selected_repo].repo_overview)

                # Display repo-specific chat history
                for message in st.session_state.repo_chat_histories[selected_repo][-10:]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if "analysis" in message:
                            with st.expander("Analysis Details"):
                                st.write(message["analysis"])

                # Repo-specific chat input
                if prompt := st.chat_input(f"Ask about {selected_repo}"):
                    with st.chat_message("user"):
                        st.write(prompt)
                    st.session_state.repo_chat_histories[selected_repo].append({"role": "user", "content": prompt})

                    with st.chat_message("assistant"):
                        # Process query for single repository
                        repo_analyzer = st.session_state.repositories[selected_repo]
                        response = analyzer_agent.process_single_repo_query(prompt, repo_analyzer)
                        st.write(response['answer'])
                        if "analysis" in response:
                            with st.expander("Analysis Details"):
                                st.write(response["analysis"])

                    st.session_state.repo_chat_histories[selected_repo].append({
                        "role": "assistant",
                        "content": response['answer'],
                        "analysis": response.get("analysis", "")
                    })
                    # Keep only the most recent 5 interactions (10 messages) in the repo-specific chat history
                    if len(st.session_state.repo_chat_histories[selected_repo]) > 10:
                        st.session_state.repo_chat_histories[selected_repo] = st.session_state.repo_chat_histories[
                                                                                  selected_repo][-10:]

            else:
                st.info("Please select a repository to start chatting.")



        else:
            st.write("Generator is not connected yet")

    # Make sure to add this function to your imports
    if 'repo_chat_histories' not in st.session_state:
        st.session_state.repo_chat_histories = {}


    with tab3:
        st.title("Repository Workflow Analyzer")
        if selected_repo:
            if st.session_state.get('show_analysis', False):
                if selected_repo in st.session_state.analysis_results:
                    st.write(st.session_state.analysis_results[selected_repo])
                    analyzer_agent.functionality_extractor(st.session_state.analysis_results[selected_repo])
                else:
                    st.info("Analysis not yet run. Use the 'Run Workflow Analysis' button in the sidebar.")

            if st.session_state.get('show_functionalities', False):
                analyzer_agent.json_reader()
        else:
            st.info("Please select a repository and choose analysis options in the sidebar.")




if __name__ == "__main__":
    main()
