# 🧑‍💻 Contributing to ZenML Projects

A big welcome and thank you for considering contributing to ZenML Projects! It’s people
like you that make it a reality for users
in our community.

Reading and following these guidelines will help us make the contribution
process easy and effective for everyone
involved. It also communicates that you agree to respect the developers' time
management and develop these open-source projects. In return, we will reciprocate that respect by reading your
issue(s), assessing changes, and helping
you finalize your pull requests.

## ⚡️ Quicklinks

* [Code of Conduct](#-code-of-conduct)
* [Getting Started](#-getting-started)
    * [Issues](#-issues)
    * [Pull Requests](#-pull-requests)
* [Getting Help](#-getting-help)

## 🧑‍⚖️ Code of Conduct

We take our open-source community seriously and hold ourselves and other
contributors to high standards of communication.
By participating and contributing to this project, you agree to uphold
our [Code of Conduct](https://github.com/zenml-io/zenml-projects/blob/master/CODE-OF-CONDUCT.md).

## 🛫 Getting Started

Contributions are made to this repo via Issues and Pull Requests (PRs). A few
general guidelines that cover both:

- To report security vulnerabilities, please get in touch
  at [support@zenml.io](mailto:support@zenml.io), monitored by
  our security team.
- Search for existing Issues and PRs before creating your own.
- We work hard to make sure issues are handled on time, but it could take a
  while to investigate the root cause depending on the impact.

A friendly ping in the comment thread to the submitter or a contributor can help
draw attention if your issue is blocking.

The best way to start is to check the
[`good-first-issue`](https://github.com/zenml-io/zenml-dashboard/labels/good%20first%20issue)
label on the issue board. The core team creates these issues as necessary
smaller tasks that you can work on to get deeper into ZenML dashboard internals. These
should generally require relatively simple changes, probably affecting just one
or two files which we think are ideal for people new to ZenML dashboard.

The next step after that would be to look at the
[`good-second-issue`](https://github.com/zenml-io/zenml-dashboard/labels/good%20second%20issue)
label on the issue board. These are a bit more complex, might involve more
files, but should still be well-defined and achievable to people relatively new
to ZenML dashboard.

### ⁉️ Issues

Issues should be used to report problems with the library, request a new
feature, or to discuss potential changes before
a PR is created. When you create a new issue, please use one of the provided
templates. These templates will guide you through collecting and
providing the information we need to investigate.

If you find an Issue that addresses your problem, please add your own
reproduction information to the
existing issue rather than creating a new one. Adding
a [reaction](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/)
can also help by
indicating to our maintainers that a particular issue is affecting more than
just the reporter.

### 🏷 Pull Requests

Pull Requests (PRs) to ZenML Projects are always welcome. In
general, PRs should:

- Only fix/add the functionality in question **OR** address widespread
  whitespace/style issues, not both.
- Add unit or integration tests for fixed or changed functionality (if a test
  suite already exists).
- Address a single concern in the least number of changed lines as possible.
- Include documentation in the repo or in your Pull Request.
- Be accompanied by a filled-out pull request template (loaded automatically
  when a PR is created). This helps reviewers understand the motivation and
  context for your changes.

For changes that address core functionality or would require breaking changes, it's best to open
an Issue to discuss your proposal first. This is not required but can save time
creating and reviewing changes.

In general, we follow
the ["fork-and-pull" Git workflow](https://github.com/susam/gitpr)

1. Review and sign
   the [Contributor License Agreement](https://cla-assistant.io/zenml-io/zenml-dashboard) (
   CLA).
2. Fork the repository to your own GitHub account.
3. Clone the project to your machine.
4. Checkout the **main** branch <- `git checkout main`.
5. Create a branch locally off the **main** branch with a succinct but descriptive name.
6. Commit changes to the branch.
7. Format your code by running `bash scripts/format.sh` before committing.
8. Push changes to your fork.
9. Open a PR in our repository to the `main` branch and
   follow the PR template so that we can efficiently review the changes.

#### Code Formatting

All code must pass our formatting checks before it can be merged. We use [ruff](https://github.com/astral-sh/ruff) for code formatting and linting.

To format your code locally:
```bash
# Run from the project root
bash scripts/format.sh
```

Our CI pipeline automatically checks if your code is properly formatted. If the check fails, you'll need to run the formatting script locally and commit the changes before your PR can be merged.

### 🚨 Reporting a Vulnerability

If you think you have found a vulnerability, and even if you are not sure about it,
please report it right away by sending an
email to: support@zenml.com. Please try to be as explicit as possible,
describing all the steps and example code to
reproduce the security issue.

We will review it thoroughly and get back to you.

Please refrain from publicly discussing a potential security vulnerability as
this could potentially put our users at
risk! It's better to discuss privately and give us a chance to find a solution
first, to limit the potential impact
as much as possible.


## 🆘 Getting Help

Join us in the [ZenML Slack Community](https://zenml.io/slack-invite/) to 
interact directly with the core team and community at large. This is a good 
place to ideate, discuss concepts or ask for help.