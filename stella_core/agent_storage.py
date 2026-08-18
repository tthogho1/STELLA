import os
import importlib.util
import traceback

from stella_core.models.agent import Agent


class AgentStorage:
    """
    Loads and stores information about all agents found under the given directories,
    including their subdirectories.
    """
    agents = {}

    def __init__(self, agent_dirs):
        """
        :param agent_dirs: Directories to recursively scan for Agent subclasses. The core
                           runtime has no built-in notion of "the" agents directory -- the
                           host application (e.g. app/server.py) decides which directories
                           make up its agent set and passes them in.
        """
        self.agent_dirs = list(agent_dirs)
        self._load_agents()

    def _load_agents(self):
        print("Loading agents...")

        for agents_dir in self.agent_dirs:
            if not os.path.isdir(agents_dir):
                print(f"[AgentStorage] !! Skipping agent directory {agents_dir}, it does not exist")
                continue
            self._load_agents_recursive(agents_dir)

        print(f"Loaded agents: {self.agents}")

    def _load_agents_recursive(self, directory):
        """ Recursively load agents from the given directory and its subdirectories. """
        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)

            if os.path.isdir(full_path):
                # If it's a directory, recursively load agents from it
                self._load_agents_recursive(full_path)
            elif entry.endswith('.py') and entry != '__init__.py':
                # Load the agent from the Python file
                self._load_agent_from_file(full_path, entry)

    def _load_agent_from_file(self, file_path, filename):
        """
        Load an agent from a Python file.

        Agents are third-party code dropped into a directory, and one of them failing to
        import -- a missing optional dependency is the usual reason -- must not stop the
        server from starting with the rest. The file is skipped with an explanation
        instead.
        """
        # Generate a module name based on the filename
        module_name = filename[:-3]

        # Load and import the module
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[AgentStorage] !! Skipping {filename}, it could not be imported "
                  f"({type(e).__name__}: {e})")
            traceback.print_exc()
            return

        # Iterate over all attributes in the module
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            # If the attribute is a class, is a subclass of Agent, and isn't Agent itself
            if isinstance(attribute, type) and issubclass(attribute, Agent) and attribute != Agent:
                # Instantiate the agent and add to the storage
                try:
                    agent_instance = attribute()
                except Exception as e:
                    # Every agent class has to be constructible with no arguments.
                    print(f"[AgentStorage] !! Skipping {attribute_name} in {filename}, it could "
                          f"not be constructed ({type(e).__name__}: {e})")
                    traceback.print_exc()
                    continue

                if agent_instance.agent_id in self.agents:
                    print(f"[AgentStorage] !! Duplicate agent_id '{agent_instance.agent_id}' "
                          f"in {filename}; the previous one is being replaced")
                self.agents[agent_instance.agent_id] = agent_instance

    def load(self, agent_id: str):
        return self.agents.get(agent_id, None)

    def reload(self):
        self.agents = {}
        self._load_agents()

    def __getitem__(self, key: str) -> 'Agent':
        return self.agents[key]

    def __iter__(self):
        return iter(self.agents.values())

    def __len__(self):
        return len(self.agents)
