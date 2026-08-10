# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional README with comparison table and quickstart guide
- CONTRIBUTING.md with development setup guidelines
- Issue templates (bug report, feature request) with RAG-specific fields
- Architecture documentation with module breakdown and extension points

## [0.1.0] - 2026-08-09

### Added
- Core inspection engine with 7 RAG architecture support
- Retrieval analysis layer (similarity scoring, chunk classification)
- Context construction analysis (token efficiency, deduplication)
- Generation grounding analysis (hallucination detection, citation tracking)
- RAG Health Score (weighted: relevance 35% + efficiency 35% + grounding 30%)
- Terminal reporter with colored tables and progress bars
- JSON output mode
- 4 inspection modes: full, retrieval, context, quick
- Architecture-specific diagnostic recommendations
- TOML pipeline configuration
- Token counting via tiktoken BPE tokenizer
