"""Posts a passing `head-approved` check when a pull request head is approved.

`.github/workflows/head-approved.yml` runs this script and gives the inputs
through environment variables. Read the security rules in that file before you
change this script.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

CHECK_NAME = "head-approved"
WORKFLOW_FILE = "head-approved.yml"

REPO = os.environ["REPO"]
DEFAULT_BRANCH = os.environ["DEFAULT_BRANCH"]
APP_TOKEN = os.environ["APP_TOKEN"]
APP_SLUG = os.environ["APP_SLUG"]
READ_TOKEN = os.environ["READ_TOKEN"]


def api(token, path, params=None, body=None):
    """Sends a request to the GitHub REST API and returns the JSON response.

    Sends a POST request if `body` is given, otherwise a GET request.
    """
    url = f"https://api.github.com/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def quote(value):
    """Encodes `value` for use as one segment of an API path."""
    return urllib.parse.quote(value, safe="")


def has_write(login):
    """Returns true if the user has write access or higher."""
    try:
        permission = api(
            APP_TOKEN, f"repos/{REPO}/collaborators/{quote(login)}/permission"
        )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    return permission["permission"] in ("admin", "write")


def is_trusted(login):
    """Returns true if the user is a trusted actor."""
    if login == "mergify[bot]":
        return True
    if login.endswith("[bot]"):
        return False
    return has_write(login)


def approved_at(pr, sha):
    """Returns true if a user with write access approved commit `sha` of pull request `pr`."""
    approvers = set()
    page = 1
    while True:
        reviews = api(
            READ_TOKEN,
            f"repos/{REPO}/pulls/{pr}/reviews",
            {"per_page": 100, "page": page},
        )
        for review in reviews:
            if (
                review["state"] == "APPROVED"
                and review["commit_id"] == sha
                and review["user"]
            ):
                approvers.add(review["user"]["login"])
        if len(reviews) < 100:
            break
        page += 1
    return any(has_write(login) for login in sorted(approvers))


def passed(sha):
    """Returns true if this app posted a passing check on the commit."""
    runs = api(
        READ_TOKEN,
        f"repos/{REPO}/commits/{quote(sha)}/check-runs",
        {"check_name": CHECK_NAME, "filter": "all", "per_page": 100},
    )
    return any(
        (run.get("app") or {}).get("slug") == APP_SLUG
        and run["conclusion"] == "success"
        for run in runs["check_runs"]
    )


def will_pass(sha):
    """Returns true if this app posts a passing check on the commit.

    Waits while a run of this workflow for the commit is not complete, because
    that run decides the result.
    """
    while True:
        if passed(sha):
            return True
        runs = api(
            READ_TOKEN,
            f"repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs",
            {"event": "pull_request_target", "head_sha": sha},
        )
        if all(run["status"] == "completed" for run in runs["workflow_runs"]):
            return passed(sha)
        time.sleep(10)


def in_default_branch(sha):
    """Returns true if the commit is in the default branch."""
    comparison = api(
        READ_TOKEN, f"repos/{REPO}/compare/{quote(DEFAULT_BRANCH)}...{quote(sha)}"
    )
    return comparison["status"] in ("behind", "identical")


def parents_ok(sha):
    """Returns true if each parent of the commit passes or is in the default branch."""
    commit = api(READ_TOKEN, f"repos/{REPO}/commits/{quote(sha)}")
    return all(
        in_default_branch(parent["sha"]) or will_pass(parent["sha"])
        for parent in commit["parents"]
    )


def post(sha, title, summary):
    """Posts a passing check on the commit."""
    print(f"{sha}: {title}")
    api(
        APP_TOKEN,
        f"repos/{REPO}/check-runs",
        body={
            "name": CHECK_NAME,
            "head_sha": sha,
            "conclusion": "success",
            "output": {"title": title, "summary": summary},
        },
    )


def on_pull_request_target():
    """Evaluates the head commit after a pull request is opened or pushed to."""
    action = os.environ["ACTION"]
    pr = os.environ["PR"]
    head_sha = os.environ["HEAD_SHA"]
    before = os.environ["BEFORE"]
    sender = os.environ["SENDER"]

    if os.environ["PR_AUTHOR"] == "mergify[bot]" and os.environ["HEAD_REPO"] == REPO:
        # A merge queue head merges queued pull request heads into the default branch.
        if parents_ok(head_sha):
            post(
                head_sha,
                "Merge queue",
                f"Each parent of this merge queue commit passed or is in {DEFAULT_BRANCH}.",
            )
    elif approved_at(pr, head_sha):
        post(head_sha, "Approved", "A user with write access approved this commit.")
    elif (
        action == "synchronize"
        and is_trusted(sender)
        and (approved_at(pr, before) or will_pass(before))
    ):
        post(
            head_sha,
            "Pushed by a trusted actor",
            f"{sender} pushed this commit, and the previous head commit passed.",
        )


def on_workflow_run():
    """Evaluates the head commits of the pull request that got a review."""
    head = f"{os.environ['RUN_HEAD_OWNER']}:{os.environ['RUN_HEAD_BRANCH']}"
    for pull in api(READ_TOKEN, f"repos/{REPO}/pulls", {"state": "open", "head": head}):
        sha = pull["head"]["sha"]
        if approved_at(pull["number"], sha):
            post(sha, "Approved", "A user with write access approved this commit.")


if os.environ["EVENT"] == "pull_request_target":
    on_pull_request_target()
else:
    on_workflow_run()
