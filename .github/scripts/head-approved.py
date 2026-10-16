"""Posts a passing `head-approved` check when a pull request head is approved.

`.github/workflows/head-approved.yml` runs this script and gives the inputs
through environment variables. Read the security rules in that file before you
change this script.
"""

import functools
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


@functools.cache
def git_commit(sha):
    """Returns the tree and parents of the commit."""
    return api(READ_TOKEN, f"repos/{REPO}/git/commits/{quote(sha)}")


@functools.cache
def mergify_check_heads():
    """Returns the head commits of the open mergify check pull requests."""
    heads = []
    page = 1
    while True:
        pulls = api(
            READ_TOKEN,
            f"repos/{REPO}/pulls",
            {"state": "open", "per_page": 100, "page": page},
        )
        for pull in pulls:
            repo = pull["head"]["repo"]
            if (
                pull["user"]["login"] == "mergify[bot]"
                and repo
                and repo["full_name"] == REPO
            ):
                heads.append(pull["head"]["sha"])
        if len(pulls) < 100:
            break
        page += 1
    return heads


def same_tree_passes(sha, exclude):
    """Returns true if a merge queue head other than `exclude` has the same tree
    as the commit and passes.

    Only the check on the matching commit is trusted. Merge queue heads are
    searched because Mergify builds stacked merge queue heads on copies of the
    merge queue heads ahead of them.
    """
    tree = git_commit(sha)["tree"]["sha"]
    matches = [
        head
        for head in mergify_check_heads()
        if head not in (sha, exclude) and git_commit(head)["tree"]["sha"] == tree
    ]
    return any(passed(head) for head in matches) or any(
        will_pass(head) for head in matches
    )


@functools.cache
def commit_ok(sha, exclude, depth):
    """Check an individual commit for `mergify_check_parents_ok`

    Returns true if any of following are true:
     - The commit passes head approval
     - The commit is in the default branch
     - The commit has the same tree as a commit that passes head approval, and
       each of its parents is ok.

    `exclude` is the mergify check PR head being evaluated. `depth` limits the
    recursion through parents.
    """
    if in_default_branch(sha) or will_pass(sha):
        return True
    # `not same_tree_passes(...)` is the tricky case. Mergify constructs each
    # check branch from scratch, so there won't be a head approval job for the
    # exact commit SHA of previous merges in the stack. But there should be a
    # head approval job of some other SHA with the same tree, so we can look
    # for that.
    if depth == 0 or not same_tree_passes(sha, exclude):
        return False
    return all(
        commit_ok(parent["sha"], exclude, depth - 1)
        for parent in git_commit(sha)["parents"]
    )


def mergify_check_parents_ok(sha):
    """Determines head approval for pull requests opened by Mergify to check queued PRs.

    These PRs run CI on a combination of up to 5 PRs in the merge queue.
    Head approval requires that each of the PRs involved have head approval.

    Limitations:

    - A stacked mergify check passes only if the mergify checks ahead of
      it are still open when this runs. If Mergify closes one of them and then
      pushes this head again, this head does not pass.
    - If two mergify checks have the same tree (for example, because of an
      empty pull request in the queue), each can wait for the other until the
      job times out, and neither passes.
    """
    max_depth = 5
    return all(
        commit_ok(parent["sha"], sha, max_depth)
        for parent in git_commit(sha)["parents"]
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
        if mergify_check_parents_ok(head_sha):
            post(
                head_sha,
                "Merge queue",
                "Each constituent of this mergify check PR passed.",
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
    """Evaluates the head commit of the pull request that got a review."""
    # The run's `head_repository` is the base repository, even for pull
    # requests from forks, so find the pull request by branch and commit.
    branch = os.environ["RUN_HEAD_BRANCH"]
    run_sha = os.environ["RUN_HEAD_SHA"]
    page = 1
    while True:
        pulls = api(
            READ_TOKEN,
            f"repos/{REPO}/pulls",
            {"state": "open", "per_page": 100, "page": page},
        )
        for pull in pulls:
            sha = pull["head"]["sha"]
            if (
                pull["head"]["ref"] == branch
                and sha == run_sha
                and approved_at(pull["number"], sha)
            ):
                post(sha, "Approved", "A user with write access approved this commit.")
        if len(pulls) < 100:
            break
        page += 1


def on_workflow_dispatch():
    """Evaluates the head commit of the pull request given as input."""
    pr = int(os.environ["DISPATCH_PR"])
    pull = api(READ_TOKEN, f"repos/{REPO}/pulls/{pr}")
    if pull["state"] != "open":
        print(f"Pull request {pr} is not open.")
        return
    sha = pull["head"]["sha"]
    if approved_at(pr, sha):
        post(sha, "Approved", "A user with write access approved this commit.")
    else:
        print(f"{sha}: No approval from a user with write access.")


event = os.environ["EVENT"]
if event == "pull_request_target":
    on_pull_request_target()
elif event == "workflow_run":
    on_workflow_run()
elif event == "workflow_dispatch":
    on_workflow_dispatch()
else:
    raise ValueError(f"Unexpected event: {event}")
