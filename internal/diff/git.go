package diff

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"
)

func Git(dirPath string) (string, error) {
	// ensure git is installed and available to run
	gitFilePath, lookErr := exec.LookPath("git")
	if lookErr != nil {
		return "", fmt.Errorf("git not found: %w", lookErr)
	}

	// get the diff
	var cmd = exec.Command(gitFilePath, "diff",
		"--cached",                 // show all staged changes or changes between the index and the working tree
		"--ignore-submodules=all",  // ignore changes to submodules
		"--diff-algorithm=minimal", // use the minimal diff algorithm
		"--no-ext-diff",            // do not use external diff helper
		"--ignore-all-space",       // ignore whitespace when comparing lines
		"--ignore-blank-lines",     // ignore changes whose lines are all blank
		"--no-color",               // do not use any color in the output
		"--patch",                  // generate patch (unified diff) format
		"--",
		":(exclude)*.sum",  // exclude .sum files
		":(exclude)*.lock", // exclude .lock files
		":(exclude)*.log",  // exclude .log files
		":(exclude)*.out",  // exclude .out files
		":(exclude)*.tmp",  // exclude .tmp files
		":(exclude)*.bak",  // exclude .bak files
		":(exclude)*.swp",  // exclude .swp files
		":(exclude)*.env",  // exclude .env files
	)

	cmd.Dir = dirPath

	var stdOut, stdErr bytes.Buffer

	stdOut.Grow(1024 * 8) //nolint:mnd // 8KB

	cmd.Stdout = &stdOut
	cmd.Stderr = &stdErr

	if err := cmd.Run(); err != nil {
		if stdErr.Len() > 0 {
			var lines = strings.Split(stdErr.String(), "\n")

			// remove empty lines
			for i := 0; i < len(lines); i++ {
				if len(strings.TrimSpace(lines[i])) == 0 {
					lines = append(lines[:i], lines[i+1:]...)
					i--
				}
			}

			err = fmt.Errorf("%s: %w", strings.Join(lines, "; "), err)
		}

		return "", fmt.Errorf("git diff failed: %w", err)
	}

	return stdOut.String(), nil
}
