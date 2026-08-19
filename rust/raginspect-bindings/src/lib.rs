#![allow(clippy::useless_conversion)]

extern crate raginspect as raginspect_lib;

use ::raginspect::classifier::ArchitectureClassifier;
use ::raginspect::topology::TopologyAnalyzer;
use ::raginspect::{InspectMode, Inspector, PipelineConfig, RagArchitecture};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert a Python dict -> serde_json::Value merged with PipelineConfig::default() -> PipelineConfig.
fn dict_to_config(py: Python<'_>, pipeline: &Bound<'_, PyDict>) -> PyResult<PipelineConfig> {
    let json_str: String = py
        .import_bound("json")?
        .call_method1("dumps", (pipeline,))?
        .extract()?;

    let value: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse pipeline dict: {e}")))?;

    let mut default_val = serde_json::to_value(PipelineConfig::default())
        .map_err(|e| PyValueError::new_err(format!("Default config error: {e}")))?;

    if let (serde_json::Value::Object(ref mut default_map), serde_json::Value::Object(map)) =
        (&mut default_val, value)
    {
        for (k, v) in map {
            if let serde_json::Value::Object(nested) = v {
                if let Some(serde_json::Value::Object(default_nested)) = default_map.get_mut(&k) {
                    for (nk, nv) in nested {
                        default_nested.insert(nk, nv);
                    }
                } else {
                    default_map.insert(k, serde_json::Value::Object(nested));
                }
            } else {
                default_map.insert(k, v);
            }
        }
    }

    serde_json::from_value(default_val)
        .map_err(|e| PyValueError::new_err(format!("Invalid pipeline configuration: {e}")))
}

/// Maps string representation to InspectMode enum.
pub fn parse_inspect_mode(s: &str) -> Result<InspectMode, String> {
    match s.to_lowercase().as_str() {
        "full" => Ok(InspectMode::Full),
        "retrieval" => Ok(InspectMode::Retrieval),
        "context" => Ok(InspectMode::Context),
        "quick" => Ok(InspectMode::Quick),
        _ => Err(format!(
            "Unknown inspect_mode: '{}'. Valid modes: full, retrieval, context, quick",
            s
        )),
    }
}

/// Maps string representation to RagArchitecture enum.
pub fn parse_architecture(s: &str) -> PyResult<RagArchitecture> {
    match s.to_lowercase().as_str() {
        "naive" => Ok(RagArchitecture::Naive),
        "advanced" => Ok(RagArchitecture::Advanced),
        "modular" => Ok(RagArchitecture::Modular),
        "agentic" => Ok(RagArchitecture::Agentic),
        "graph" => Ok(RagArchitecture::Graph),
        "hyde" => Ok(RagArchitecture::Hyde),
        "multimodal" => Ok(RagArchitecture::Multimodal),
        _ => Err(PyValueError::new_err(format!(
            "Unknown architecture: '{}'. Valid architectures: naive, advanced, modular, agentic, graph, hyde, multimodal",
            s
        ))),
    }
}

#[pyfunction]
#[allow(clippy::useless_conversion)]
#[pyo3(signature = (pipeline, query="What is RAG?", inspect_mode="full", architecture="naive"))]
fn profile(
    py: Python<'_>,
    pipeline: &Bound<'_, PyDict>,
    query: &str,
    inspect_mode: &str,
    architecture: &str,
) -> PyResult<PyObject> {
    let config = dict_to_config(py, pipeline)?;

    let mode_enum = parse_inspect_mode(inspect_mode).map_err(PyValueError::new_err)?;
    let arch_enum = parse_architecture(architecture)?;

    let mut inspector = Inspector::new(config, None);
    inspector.set_architecture(arch_enum);
    let report = inspector
        .inspect(query, mode_enum)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let report_val = serde_json::to_value(&report)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize report: {}", e)))?;
    let report_json = report_val.to_string();
    let py_dict = py
        .import_bound("json")?
        .call_method1("loads", (report_json,))?;

    Ok(py_dict.into())
}

#[pyfunction]
#[allow(clippy::useless_conversion)]
fn classify(py: Python<'_>, pipeline: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let config = dict_to_config(py, pipeline)?;
    let topology = TopologyAnalyzer::new().analyze(&config);
    let result = ArchitectureClassifier::new().classify(&topology);

    let res_val = serde_json::to_value(&result)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize result: {}", e)))?;
    let res_json = res_val.to_string();
    let py_dict = py
        .import_bound("json")?
        .call_method1("loads", (res_json,))?;

    Ok(py_dict.into())
}

#[pymodule]
fn raginspect(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(profile, m)?)?;
    m.add_function(wrap_pyfunction!(classify, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_config_conversion() {
        let val = serde_json::json!({});
        let mut default_val = serde_json::to_value(PipelineConfig::default()).unwrap();
        if let (serde_json::Value::Object(ref mut default_map), serde_json::Value::Object(map)) =
            (&mut default_val, val)
        {
            for (k, v) in map {
                default_map.insert(k, v);
            }
        }
        let config: Result<PipelineConfig, _> = serde_json::from_value(default_val);
        assert!(
            config.is_ok(),
            "Minimal config serde_json::Value should convert to PipelineConfig successfully"
        );
        let cfg = config.unwrap();
        assert_eq!(cfg.name, "Default-RAG-Pipeline");
    }

    #[test]
    fn test_unknown_inspect_mode_error() {
        let res = parse_inspect_mode("invalid_mode");
        assert!(
            res.is_err(),
            "Unknown inspect_mode string should return an error"
        );
        assert!(res.unwrap_err().contains("Unknown inspect_mode"));
    }

    #[test]
    fn test_valid_inspect_modes() {
        assert_eq!(parse_inspect_mode("full").unwrap(), InspectMode::Full);
        assert_eq!(
            parse_inspect_mode("retrieval").unwrap(),
            InspectMode::Retrieval
        );
        assert_eq!(parse_inspect_mode("context").unwrap(), InspectMode::Context);
        assert_eq!(parse_inspect_mode("quick").unwrap(), InspectMode::Quick);
    }

    #[test]
    fn test_architecture_classifier_default() {
        let config = PipelineConfig::default();
        let topology = TopologyAnalyzer::new().analyze(&config);
        let result = ArchitectureClassifier::new().classify(&topology);
        assert_eq!(result.architecture, RagArchitecture::Naive);
        assert!(result.confidence >= 0.9);
    }

    #[test]
    fn test_parse_architecture() {
        assert!(parse_architecture("bogus").is_err());
        assert_eq!(
            parse_architecture("advanced").unwrap(),
            RagArchitecture::Advanced
        );
    }
}
