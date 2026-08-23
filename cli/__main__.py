import json
import os
import argparse

from cli.client.stella_client import StellaClient
from cli.shell import Shell

from cli.commands import ServeCommand, ConfigureCommand


def setup_parser():
    parser = argparse.ArgumentParser(description="Stella Command Line Interface")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    ServeCommand.init_parser(subparsers)
    ConfigureCommand.init_parser(subparsers)

    return parser


def main():
    # Load config
    with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
        config = json.load(f)

    # Create client
    client = StellaClient(
        host=config['host'],
        port=config.get('port'),
        # StellaClient has always taken this; nothing passed it, so the CLI could only
        # ever speak plain http -- including the password on /login and the JWT on every
        # request after it. Optional so an existing config.json without the key still
        # loads, and omitting "port" is allowed for a proxy on the standard port.
        ssl=config.get('ssl', False),
    )

    # Setup parser and parse arguments (if any)
    parser = setup_parser()
    args = parser.parse_args()

    # If there's a command, execute it, otherwise start the shell
    if args.command:
        command = args.command_class(client, args)
        command.execute()
    else:
        shell = Shell(client)
        shell.start()


if __name__ == '__main__':
    main()