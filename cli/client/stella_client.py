import os

from .cli_design import *
import socketio
import json
import time
import requests
import zipfile
import io


class Session:
    def __init__(self, session_file_path=None):
        self.session_file_path = session_file_path
        self.access_token = None
        self.workspace_id = None
        self.chat_id = None
        self.chat_connection_string = None
        self.chat_message_string = None

        self.load_session()

    def load_session(self):
        # If a session file is not provided, create it in the current directory
        if self.session_file_path is None:
            self.session_file_path = os.path.join(os.path.dirname(__file__), "../session.json")

        # If the session file does not exist, create it
        if not os.path.exists(self.session_file_path):
            with open(self.session_file_path, "w+") as file:
                json.dump({}, file)

        try:
            # Load the session file
            with open(self.session_file_path, "r") as file:
                session_dict = json.load(file)
                self.access_token = session_dict.get("access_token", None)
                self.workspace_id = session_dict.get("workspace_id", None)
                self.chat_id = session_dict.get("chat_id", None)
                self.chat_connection_string = session_dict.get("chat_connection_string", None)
                self.chat_message_string = session_dict.get("chat_message_string", None)
        except Exception as e:
            print_error(f"(Session error: {e})")
            print_info(f"Could not load session file. Generating a new one.")
            return {}

    def save_session(self):
        with open(self.session_file_path, "w+") as file:
            json.dump(self.to_dict(), file, indent=4)

    def to_dict(self):
        return {
            "access_token": self.access_token,
            "workspace_id": self.workspace_id,
            "chat_id": self.chat_id,
            "chat_connection_string": self.chat_connection_string,
            "chat_message_string": self.chat_message_string,
        }


class StellaClient:
    def __init__(self, host=None, port=None, session_file_path=None, ssl=False):
        self.spinner = Spinner()
        self.host = host
        self.port = port
        self.ssl = ssl
        self.socket_url = self.compose_url("chat/connect")
        self.socketio_namespace = "/chat"

        self.session = Session(session_file_path=session_file_path)

        self.sio = socketio.Client()
        self.sio.on('connect', self.on_connect, namespace=self.socketio_namespace)
        self.sio.on('message', self.on_message, namespace=self.socketio_namespace)
        self.sio.on('chat_information', self.on_chat_information, namespace=self.socketio_namespace)
        self.sio.on('disconnect', self.on_disconnect, namespace=self.socketio_namespace)

        self.should_wait_for_response = True
        self.waiting_for_response = False
        self.initial_message = None

    @staticmethod
    def access_error(status_code):
        """
        Turns a rejected request into something the user can act on.

        401 and 403 need different remedies: 401 means the token is missing or expired,
        403 means it is fine but this account does not own what was asked for -- which is
        what a stale chat_id in session.json looks like after switching user.
        """
        if status_code == 403:
            return "This account does not have access to that. Try /workspace list, or /login as another user."
        return "You are not authenticated. Please login."

    def auth_headers(self, json_body=True):
        """
        The Authorization header, and by default the JSON content type with it.

        A multipart upload has to pass json_body=False: requests writes its own
        Content-Type with the multipart boundary, and a hand-written one overrides it,
        leaving the server unable to find the boundary and the file invisible.
        """
        # Check if the user is logged in
        if self.session.access_token is None:
            return {}
        headers = {"Authorization": f"Bearer {self.session.access_token}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def verify_connection(self):
        """
        Verify that a connection to the server can be established.
        :return:
        """
        try:
            response = requests.get(self.compose_url("ping"))
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            return False

    def compose_url(self, endpoint):
        """
        Builds a server URL.

        The port is omitted when there is none, so an https deployment behind a reverse
        proxy on 443 works: the old form interpolated None into the host and produced
        "https://example.comNone/ping".
        """
        scheme = "https" if self.ssl else "http"
        authority = f"{self.host}:{self.port}" if self.port else f"{self.host}"
        return f"{scheme}://{authority}/{endpoint}"

    def login(self, username, password):
        try:
            response = requests.post(
                self.compose_url("auth/login"),
                json={"username": username, "password": password}
            )
            if response.status_code != 200:
                print_error("Login failed, please try again. (Wrong username or password)")
                return None
            access_token = response.json()["access_token"]
            print_success("Login successful.")

            # Save access token to session
            self.session.access_token = access_token
            self.session.save_session()

            return access_token
        except Exception as e:
            print_error(f"Login failed. Please try again. ({e})")

    def connect_to_workspace(self, workspace_id):
        response = requests.get(self.compose_url(f"workspace/{workspace_id}"), headers=self.auth_headers())
        if response.status_code == 500:
            print_error(f"Workspace with ID {workspace_id} not found.")
        elif response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            print(response.text)
            print_error("Failed to connect to workspace ({}).".format(response.status_code))
        else:
            self.session.workspace_id = workspace_id
            # Create a new chat
            self.create_chat(workspace_id)
            self.session.save_session()
            self.connect_to_chat()
            print_success("Successfully connected to workspace.")

    def install_agent(self, package_name, version=None):
        # Download and install the agent
        try:
            response = requests.get(self.compose_url(f"agent/download"), headers=self.auth_headers(),
                                    params={"query": package_name, "version": version})
        except Exception as e:
            print_error(f"Failed to download package: {package_name}:{version}, {e}")
            return

        if version is None:
            version = "latest"

        if response.status_code == 200:
            print_success(f"Successfully installed {package_name}:{version}")
        elif response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code == 404:
            print_info(f"Package not found: {package_name}:{version}")
        else:
            print_error(f"Failed to download package: {package_name}:{version}, {response.text}")

        # Reload the server's agent storage
        print_info(f"Reloading available agents...")
        try:
            response = requests.get(self.compose_url(f"agent/reload"), headers=self.auth_headers())
            if response.status_code == 200:
                print_success(f"Successfully reloaded agents.")
            elif response.status_code in (401, 403):
                print_error(self.access_error(response.status_code))
            else:
                print_error(f"Failed to reload agents. ({response.text})")
        except Exception as e:
            print_error(f"Failed to reload agents. ({e})")

    def upload_source(self, archive_path):
        """
        Sends a zip of source code for the connected workspace to analyse.

        This is how code reaches a server that is not running on the machine holding it.
        The upload replaces whatever the workspace had before.
        """
        if self.session.workspace_id is None:
            print_error("You are not connected to a workspace.")
            return

        path = os.path.expanduser(archive_path)
        if not os.path.isfile(path):
            print_error(f"No such file: {archive_path}")
            return
        if not zipfile.is_zipfile(path):
            print_error(f"{archive_path} is not a zip archive.")
            return

        size = os.path.getsize(path)
        print_info(f"Uploading {os.path.basename(path)} ({size // 1024} KB)...")
        try:
            with open(path, "rb") as handle:
                response = requests.post(
                    self.compose_url(f"workspace/{self.session.workspace_id}/source"),
                    headers=self.auth_headers(json_body=False),
                    files={"file": (os.path.basename(path), handle, "application/zip")})
        except Exception as e:
            print_error(f"Upload failed. ({e})")
            return

        if response.status_code == 200:
            body = response.json()
            print_success(body["msg"])
            if body.get("entries"):
                print_info(f"Top level: {', '.join(body['entries'])}")
            print_info("Paths you name in chat start from there.")
        elif response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        else:
            print_error(self._server_message(response, "Upload failed"))

    def describe_source(self):
        """What the connected workspace currently has uploaded."""
        if self.session.workspace_id is None:
            print_error("You are not connected to a workspace.")
            return
        try:
            response = requests.get(
                self.compose_url(f"workspace/{self.session.workspace_id}/source"),
                headers=self.auth_headers())
        except Exception as e:
            print_error(f"Could not read the uploaded source. ({e})")
            return

        if response.status_code != 200:
            print_error(self._server_message(response, "Could not read the uploaded source"))
            return

        body = response.json()
        if not body.get("uploaded"):
            print_info("Nothing uploaded. Agents read the server's own SPEC_SOURCE_ROOT.")
            return
        print_success(f"{body['files']} file(s), {body['bytes'] // 1024} KB.")
        if body.get("entries"):
            print_info(f"Top level: {', '.join(body['entries'])}")

    def delete_source(self):
        """Removes the uploaded source, so the workspace reads the server's tree again."""
        if self.session.workspace_id is None:
            print_error("You are not connected to a workspace.")
            return
        try:
            response = requests.delete(
                self.compose_url(f"workspace/{self.session.workspace_id}/source"),
                headers=self.auth_headers())
        except Exception as e:
            print_error(f"Could not remove the uploaded source. ({e})")
            return

        if response.status_code == 200:
            print_success(response.json()["msg"])
        else:
            print_error(self._server_message(response, "Could not remove the uploaded source"))

    def download_spec(self, directory=None):
        """
        Fetches this workspace's generated specifications and unpacks them locally.

        Unpacked rather than left as a zip because the useful end state is a Markdown
        file the user can open. Into a directory that is empty or does not exist, so a
        download never writes over work that is already there.
        """
        if self.session.workspace_id is None:
            print_error("You are not connected to a workspace.")
            return

        target = os.path.abspath(os.path.expanduser(
            directory or f"stella-spec-{self.session.workspace_id}"))
        if os.path.isdir(target) and os.listdir(target):
            print_error(f"{target} is not empty. Name an empty or new directory.")
            return
        if os.path.exists(target) and not os.path.isdir(target):
            print_error(f"{target} is not a directory.")
            return

        try:
            response = requests.get(
                self.compose_url(f"workspace/{self.session.workspace_id}/spec"),
                headers=self.auth_headers())
        except Exception as e:
            print_error(f"Download failed. ({e})")
            return

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
            return
        if response.status_code != 200:
            print_error(self._server_message(response, "Download failed"))
            return

        try:
            archive = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            print_error("The server did not send a zip archive.")
            return

        # The server built this archive, but it is still writing to the user's own disk
        # on the strength of names inside a file that arrived over the network. The same
        # check the server applies to an upload, applied in the other direction -- it
        # cannot be shared, since the CLI never imports the server.
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        for name in names:
            if name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
                print_error(f"Refused the archive: it contains an out-of-tree path ({name}).")
                return

        os.makedirs(target, exist_ok=True)
        archive.extractall(target)
        print_success(f"Wrote {len(names)} file(s) to {target}")
        for name in sorted(names):
            if name.endswith(".md"):
                print_info(f"Start with {os.path.join(target, name)}")

    @staticmethod
    def _server_message(response, fallback):
        """The server's own explanation, which for these endpoints is the useful part."""
        try:
            return response.json().get("msg") or f"{fallback}. ({response.status_code})"
        except ValueError:
            return f"{fallback}. ({response.status_code})"

    def logout(self):
        if self.session.access_token:
            self.session.access_token = None
            self.session.save_session()
            print_success("Logout successful.")
        else:
            print_info("Currently not logged in.")

    def register(self, username, password):
        try:
            response = requests.post(
                self.compose_url("register"),
                json={"username": username, "password": password}
            )
            if response.status_code != 200:
                raise Exception(response.json()['msg'])
            print_success("Registration successful.")
        except Exception as e:
            raise Exception(f"Registration failed. Please try again. ({e})")

    def change_username(self, username):
        response = requests.put(
            self.compose_url("user/username"),
            headers=self.auth_headers(),
            json={"username": username}
        )
        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            print_error(f"Failed to change username. ({response.json()['msg']})")
        else:
            print_success("Username changed successfully.")

    def change_password(self, password):
        response = requests.put(
            self.compose_url("user/password"),
            headers=self.auth_headers(),
            json={"password": password}
        )
        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            print_error(f"Failed to change password. ({response.json()['msg']})")
        else:
            print_success("Password changed successfully.")

    def create_workspace(self, name=None):
        if name:
            response = requests.post(self.compose_url("workspace"), headers=self.auth_headers(), json={"name": name})
        else:
            response = requests.post(self.compose_url("workspace"), headers=self.auth_headers())

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
            return None
        elif response.status_code != 200:
            raise Exception(
                "Request to create a new workspace failed with status code: {}, Message: {}".format(
                    response.status_code, response.text))
        workspace_id = response.json()["workspace"]["id"]
        self.session.workspace_id = workspace_id
        self.session.save_session()
        print_success(f"Workspace {'named ' + name + ' ' if name else ''}created.")
        return workspace_id

    def rename_workspace(self, name):
        response = requests.put(self.compose_url(f"workspace/{self.session.workspace_id}/rename"), headers=self.auth_headers(),
                                json={"name": name})

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
            return None
        elif response.status_code != 200:
            raise Exception(
                "Request to rename workspace failed with status code: {}, Message: {}".format(response.status_code,
                                                                                              response.text))
        print_success(f"Workspace successfully renamed to {name}.")
        return name

    def get_all_workspaces(self):
        response = requests.get(self.compose_url("workspace"), headers=self.auth_headers())

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
            return None
        elif response.status_code != 200:
            print_error("Failed to get workspaces. ({}).".format(response.status_code))
            exit(0)
        else:
            workspaces = response.json()["workspaces"]
            return workspaces

    def get_workspace_by_id(self, workspace_id=None):
        response = requests.get(self.compose_url(f"workspace/{workspace_id}"), headers=self.auth_headers())

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            raise Exception("Failed to get workspace. ({}).".format(response.status_code))
        else:
            workspace = response.json()["workspace"]
            return workspace

    def delete_workspace(self, workspace_id):
        response = requests.delete(self.compose_url(f"workspace/{workspace_id}"), headers=self.auth_headers())

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            print_error("Failed to delete workspace. ({}).".format(response.status_code))
        else:
            print_success("Workspace deleted.")

    def add_agent(self, agent_id):
        response = requests.post(self.compose_url(f"workspace/{self.session.workspace_id}/agent"), headers=self.auth_headers(),
                                 json={"agent_id": agent_id})

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            print_error(f"Failed to add agent. ({response.json()['msg']})")
        else:
            print_success(f"Agent ({agent_id}) added to current workspace successfully.")

    def remove_agent(self, agent_id):
        response = requests.delete(self.compose_url(f"workspace/{self.session.workspace_id}/agent"), headers=self.auth_headers(),
                                   json={"agent_id": agent_id})

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            print_error("Failed to remove agent. ({}).".format(response.status_code))
        else:
            print_info(f"Agent ({agent_id}) removed from current workspace successfully.")

    def set_coordinator_agent(self, agent_id):
        response = requests.put(self.compose_url(f"workspace/{self.session.workspace_id}/coordinator"), headers=self.auth_headers(),
                                json={"agent_id": agent_id})

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            print_error("Failed to set coordinator agent. ({}).".format(response.status_code))
        else:
            print_info(f"Coordinator agent set to {agent_id}.")

    def create_chat(self, workspace_id):
        response = requests.post(self.compose_url(f"chat?workspace_id={workspace_id}"), headers=self.auth_headers())

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            raise Exception(
                "Request to create a new chat failed with status code: {}, Message: {}".format(
                    response.status_code, response.text))
        chat_id = response.json()["chat"]["chat_id"]
        if chat_id is None:
            raise Exception("Chat ID is was not returned from the server. Critical error, aborting.")
        self.session.chat_id = chat_id
        return chat_id

    @staticmethod
    def on_connect():
        pass
        # print_success("Successfully connected to chat.")

    @staticmethod
    def on_disconnect():
        pass
        # print_info("Disconnected from chat.")

    def on_message(self, message):
        # Stop the spinner
        self.spinner.stop()
        # If the message is a chat message, print it
        print_with_delay(message)
        self.waiting_for_response = False

    def on_chat_information(self, message):
        """
        Handles progress updates from the server.

        Only the final answer arrives on the 'message' event, so without these the client
        sits silent for the whole agent tree. They must not go through on_message: that
        clears waiting_for_response, which would let the user type again while the
        request is still running.
        """
        try:
            data = json.loads(message) if isinstance(message, str) else message
        except (TypeError, ValueError):
            data = message
        text = data.get("message") if isinstance(data, dict) else data
        if not text:
            return
        # Anything may arrive on the wire; print_info concatenates, so coerce first.
        text = text if isinstance(text, str) else str(text)

        # The spinner owns the current line, so pause it before writing.
        was_waiting = self.waiting_for_response
        self.spinner.stop()
        print_info(text)
        if was_waiting:
            self.spinner.start()

    def get_connection_string(self, chat_id):
        response = requests.get(self.compose_url(f"chat/authorize?chat_id={chat_id}"), headers=self.auth_headers())

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            raise Exception(
                "Request to get a chat connection authorization string failed with status code: {}, Message: {}".format(
                    response.status_code, response.text))
        connection_string = response.json()["string"]
        return connection_string

    def disconnect_from_chat(self):
        self.sio.disconnect()

    def connect_to_chat(self):
        # Disconnect from chat if already connected
        self.disconnect_from_chat()

        # Connect to chat
        self.session.chat_connection_string = self.get_connection_string(self.session.chat_id)
        self.sio.connect(
            f"{self.socket_url}?chat_id={self.session.chat_id}&connection_string={self.session.chat_connection_string}",
            namespaces=[self.socketio_namespace])

        # Update session
        self.session.save_session()

    def send_message(self, message: str, chat_id=None):
        # Check if we're currently connected to a chat/workspace
        if self.session.chat_id is None or self.session.workspace_id is None:
            print_error("You are not connected to a chat or workspace.")
            return

        # If we're waiting for a response, don't send another message
        if self.waiting_for_response:
            return

        # If the chat id is not provided, use the one from the session
        chat_id = chat_id or self.session.chat_id

        # Start the spinner
        self.spinner.start()

        # Get a message authorization string from the server
        response = requests.get(
            self.compose_url(f"chat/authorize/message?chat_id={chat_id}"),
            headers=self.auth_headers()
        )

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
            self.spinner.stop()
            return
        elif response.status_code != 200:
            self.spinner.stop()
            raise Exception(
                "Request to get a message authorization string failed with status code: {}, Message: {}".format(
                    response.status_code, response.text))
        self.session.chat_message_string = response.json()["string"]

        # Send the message
        json_data = {
            "message": message,
            "chat_id": chat_id,
            "message_string": self.session.chat_message_string,
        }
        self.sio.emit("chat_message", json.dumps(json_data), namespace=self.socketio_namespace)
        self.waiting_for_response = True

    def list_agents(self):
        """
        Prints every agent the server has loaded, marking the ones already in this
        workspace, so /add has something to choose from.
        """
        response = requests.get(self.compose_url("agent"), headers=self.auth_headers())
        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
            return
        if response.status_code != 200:
            print_error(f"Could not list agents ({response.status_code}).")
            return

        agents = response.json().get("agents", [])
        if not agents:
            print_info("This server has no agents loaded.")
            return

        # Mark what the current workspace already has, so the list is actionable.
        in_workspace = set()
        if self.session.workspace_id:
            try:
                # Returns None on 401/403 rather than raising, so do not assume a dict.
                workspace = self.get_workspace_by_id(self.session.workspace_id) or {}
                in_workspace = set(workspace.get("agents", {}))
            except Exception:
                pass

        width = max(len(a["agent_id"]) for a in agents)
        print_info(f"{len(agents)} agent(s) available. '*' is already in this workspace.")
        for a in agents:
            mark = "*" if a["agent_id"] in in_workspace else " "
            line = f"  {mark} {a['agent_id']:<{width}}  {a['short_description']}"
            if a.get("delegates_to"):
                line += f"   -> {', '.join(a['delegates_to'])}"
            print(line)
        print_info("Add one with /add <agent id>.")

    def get_trace(self, task_id=None):
        """
        Fetches the execution trace of the last request in the current chat.

        Only the final answer arrives over the socket, so this is the only way to see
        afterwards which agents ran, how the tree was shaped and where the time went.
        """
        if self.session.chat_id is None:
            print_error("You are not in a chat yet.")
            return

        url = f"chat/trace?chat_id={self.session.chat_id}"
        if task_id:
            url += f"&task_id={task_id}"

        response = requests.get(self.compose_url(url), headers=self.auth_headers())
        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
            return
        if response.status_code != 200:
            try:
                print_error(response.json().get("msg", response.text))
            except Exception:
                print_error(f"Could not fetch the trace ({response.status_code}).")
            return

        print(response.json().get("rendered", ""))

    def get_user(self):
        response = requests.get(self.compose_url("user"), headers=self.auth_headers())

        if response.status_code in (401, 403):
            print_error(self.access_error(response.status_code))
        elif response.status_code != 200:
            raise Exception(
                "Request to get a user failed with status code: {}, Message: {}".format(
                    response.status_code, response.text))
        return response.json()["user"]

    def is_logged_in(self):
        """
        Attempts to fetch the current user information from the server.
        If the request fails, the user is not logged in.
        :return:
        """
        try:
            self.get_user()
            return True
        except Exception as e:
            return False

    def connect_latest(self):
        """
        Attempts to connect to the latest chat & workspace, if any.
        Returns True if the connection was successful, False otherwise.
        :return:
        """
        if self.session.access_token:
            # Attempt to fetch the user information from the server
            try:
                user = self.get_user()
            except Exception as e:
                from cli.utils.exceptions import UserNotFoundException
                raise UserNotFoundException("User not found. Please login again.".format(e))

            # If the user has a last workspace id, fetch it and connect.
            if user["last_workspace_id"]:
                try:
                    workspace = self.get_workspace_by_id(user["last_workspace_id"])
                except Exception as e:
                    return False
                self.session.workspace_id = workspace["id"]
                self.session.save_session()
                self.connect_to_workspace(workspace["id"])


def print_with_delay(text):
    words = text.split()
    for word in words:
        colored_word = VISS_GREEN + word + ENDC
        print(colored_word, end=" ", flush=True)
        time.sleep(0.02)
    print("\n")  # Move to the next line after the sentence is completed
