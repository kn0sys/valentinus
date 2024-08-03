use log::{error, info};
use serde_json::Value;

#[derive(Debug)]
enum Md2fsError {
    SerdeJsonError,
    ParseError,
    NoKeyFound,
}
/// Where clause keys
#[derive(Debug)]
enum FilterOperations {
    EqualTo,
    GreaterThanEqualTo,
    GreaterThan,
    In,
    LessThan,
    LessThanEqualTo,
    Noop,
}

impl FilterOperations {
    /// Seek and return enum for pattern matching
    fn get_enum(s: &str) -> FilterOperations {
        match s {
            "eq" => FilterOperations::EqualTo,
            "gt" => FilterOperations::GreaterThan,
            "gte" => FilterOperations::GreaterThanEqualTo,
            "in" => FilterOperations::In,
            "lt" => FilterOperations::LessThan,
            "lte" => FilterOperations::LessThanEqualTo,
            _ => FilterOperations::EqualTo,
        }
    }
}

/// Metadata filter
#[derive(Debug)]
pub struct MetadataFilter<T> {
    /// Key to filter on
    key: String,
    /// Valid json type to filter on
    value: T,
    /// Filter operations eq, gt, gte, in, lt, lte
    filter: FilterOperations,
}

impl Default for MetadataFilter<String> {
    fn default() -> Self {
        MetadataFilter {
            key: Default::default(),
            value: Default::default(),
            filter: FilterOperations::Noop,
        }
    }
}

trait FilterString {
    fn create_filter(raw: &str) -> Result<MetadataFilter<String>, Md2fsError>;
    fn eq(self, m: MetadataFilter<String>) -> bool;
}

impl FilterString for MetadataFilter<String> {
    /// Create a filter on a valid string value
    fn create_filter(raw: &str) -> Result<MetadataFilter<String>, Md2fsError> {
        let v: Result<Value, serde_json::Error> = serde_json::from_str(raw);
        if v.is_err() {
            error!("invalid json string");
            return Err(Md2fsError::SerdeJsonError);
        }
        let u_v: Value = v.map_err(|_| Md2fsError::ParseError)?;
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("could not parse string");
            return Err(Md2fsError::ParseError);
        }
        let key = match vo {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _=> String::new()
        };
        let vo2 = match vo {
            Some(v) => v[&key].as_object(),
            _=> None
        };
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_value = &u_v[&key];
            if !p_value.is_string() {
                return Err(Md2fsError::NoKeyFound);
            }
            let value: String = match p_value.as_str() {
                Some(s) => s.to_string(),
                _=> String::new(),
            };
            let filter: FilterOperations = FilterOperations::Noop;
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = match vo2 {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _=> String::new(),
        };
        let value = match vo2 {
            Some(v) => &v[&op],
            _=> &Value::String(String::new()),
        };
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_string() {
            let value = match value.as_str() {
                Some(s) => s.to_string(),
                _=> String::new(),
            };
            return Ok(MetadataFilter { key, filter, value });
        }
        Err(Md2fsError::ParseError)
    }
    fn eq(self, m: MetadataFilter<String>) -> bool {
        match self.filter {
            FilterOperations::EqualTo => self.key == m.key && self.value == m.value,
            _ => false,
        }
    }
}

impl Default for MetadataFilter<Vec<String>> {
    fn default() -> Self {
        MetadataFilter {
            key: Default::default(),
            value: Default::default(),
            filter: FilterOperations::Noop,
        }
    }
}

trait FilterStringArray {
    fn create_filter(raw: &str) -> Result<MetadataFilter<Vec<String>>, Md2fsError>;
    fn v_in(self, m: MetadataFilter<String>) -> bool;
}

impl FilterStringArray for MetadataFilter<Vec<String>> {
    /// Create a filter on a valid string value
    fn create_filter(raw: &str) -> Result<MetadataFilter<Vec<String>>, Md2fsError> {
        let v: Result<Value, serde_json::Error> = serde_json::from_str(raw);
        if v.is_err() {
            error!("invalid json string");
            return Err(Md2fsError::SerdeJsonError);
        }
        let u_v: Value = v.map_err(|_| Md2fsError::ParseError)?;
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("could not parse string");
            return Err(Md2fsError::ParseError);
        }
        let key = match vo {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _=> String::new()
        };
        let vo2 = match vo {
            Some(v) => v[&key].as_object(),
            _=> None
        };
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_array = &u_v[&key];
            if !p_array.is_array() {
                return Err(Md2fsError::NoKeyFound);
            }
            let u_array = match p_array.as_array() {
                Some(a) => a,
                _=> &Vec::new(),
            };
            let filter: FilterOperations = FilterOperations::Noop;
            // duck invalid values in the array and fail the metadata filter
            let value: Vec<String> = u_array
                .iter()
                .map(|s| String::from(s.as_str().unwrap_or_default()))
                .collect();
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = match vo2 {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _=> String::new(),
        };
        if op.is_empty() {
            return Err(Md2fsError::ParseError);
        }
        let value = match vo2 {
            Some(v) => &v[&op],
            _=> &Value::Array(Vec::new()),
        };
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_array() {
            let possible_array = match value.as_array() {
                Some(a) => a,
                _=> &Vec::new(),
            };
            // duck invalid values in the array and fail the metadata filter
            let value = possible_array
                .iter()
                .map(|s| String::from(s.as_str().unwrap_or_default()))
                .collect();
            return Ok(MetadataFilter { key, filter, value });
        }
        Err(Md2fsError::ParseError)
    }
    fn v_in(self, m: MetadataFilter<String>) -> bool {
        match self.filter {
            FilterOperations::In => self.value.contains(&m.value),
            _ => false,
        }
    }
}

impl Default for MetadataFilter<u64> {
    fn default() -> Self {
        MetadataFilter {
            key: Default::default(),
            value: 0,
            filter: FilterOperations::Noop,
        }
    }
}

trait Filteru64 {
    fn create_filter(raw: &str) -> Result<MetadataFilter<u64>, Md2fsError>;
    fn eq(self, m: MetadataFilter<u64>) -> bool;
    fn gt(self, m: MetadataFilter<u64>) -> bool;
    fn gte(self, m: MetadataFilter<u64>) -> bool;
    fn lt(self, m: MetadataFilter<u64>) -> bool;
    fn lte(self, m: MetadataFilter<u64>) -> bool;
}

impl Filteru64 for MetadataFilter<u64> {
    /// Create a filter on a valid u64 value
    fn create_filter(raw: &str) -> Result<MetadataFilter<u64>, Md2fsError> {
        let v: Result<Value, serde_json::Error> = serde_json::from_str(raw);
        if v.is_err() {
            error!("invalid json string");
            return Err(Md2fsError::SerdeJsonError);
        }
        let u_v: Value = v.map_err(|_| Md2fsError::ParseError)?;
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("could not parse string");
            return Err(Md2fsError::ParseError);
        }
        let key = match vo {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _=> String::new()
        };
        let vo2 = match vo {
            Some(v) => v[&key].as_object(),
            _=> None
        };
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_value = &u_v[&key];
            if !p_value.is_u64() {
                return Err(Md2fsError::NoKeyFound);
            }
            let value: u64 = p_value.as_u64().unwrap_or_default();
            let filter: FilterOperations = FilterOperations::Noop;
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = match vo2 {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _=> String::new(),
        };
        let value = match vo2 {
            Some(v) => &v[&op],
            _=> &Value::String(String::new()),
        };
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_u64() {
            let value = value.as_u64().unwrap_or_default();
            return Ok(MetadataFilter { key, filter, value });
        }
        Err(Md2fsError::ParseError)
    }
    fn eq(self, m: MetadataFilter<u64>) -> bool {
        match self.filter {
            FilterOperations::EqualTo => self.key == m.key && self.value == m.value,
            _ => false,
        }
    }
    fn gt(self, m: MetadataFilter<u64>) -> bool {
        match self.filter {
            FilterOperations::GreaterThan => self.key == m.key && self.value < m.value,
            _ => false,
        }
    }
    fn gte(self, m: MetadataFilter<u64>) -> bool {
        match self.filter {
            FilterOperations::GreaterThanEqualTo => self.key == m.key && self.value <= m.value,
            _ => false,
        }
    }
    fn lt(self, m: MetadataFilter<u64>) -> bool {
        match self.filter {
            FilterOperations::LessThan => self.key == m.key && self.value > m.value,
            _ => false,
        }
    }
    fn lte(self, m: MetadataFilter<u64>) -> bool {
        match self.filter {
            FilterOperations::LessThanEqualTo => self.key == m.key && self.value >= m.value,
            _ => false,
        }
    }
}

fn process_string_filter(raw_f: &str, raw_m: &str) -> bool {
    let try_filter = <MetadataFilter<std::string::String> as FilterString>::create_filter(raw_f);
    let try_meta = <MetadataFilter<std::string::String> as FilterString>::create_filter(raw_m);
    let tf = try_filter.unwrap_or_default();
    if tf.key.is_empty() {
        log::error!("could not process string filter");
        return false;
    }
    let tm = try_meta.unwrap_or_default();
    match tf.filter {
        FilterOperations::EqualTo => tf.eq(tm),
        _ => false,
    }
}

fn process_string_array_filter(raw_f: &str, raw_m: &str) -> bool {
    let try_filter = <MetadataFilter<Vec<String>> as FilterStringArray>::create_filter(raw_f);
    let try_meta = <MetadataFilter<std::string::String> as FilterString>::create_filter(raw_m);
    let tf = try_filter.unwrap_or_default();
    if tf.key.is_empty() {
        log::error!("could not process string array filter");
        return false;
    }
    let tm = try_meta.unwrap_or_default();
    match tf.filter {
        FilterOperations::EqualTo => tf.v_in(tm),
        _ => false,
    }
}

fn process_u64_filter(raw_f: &str, raw_m: &str) -> bool {
    let try_filter = <MetadataFilter<u64> as Filteru64>::create_filter(raw_f);
    let try_meta = <MetadataFilter<u64> as Filteru64>::create_filter(raw_m);
    let tf = try_filter.unwrap_or_default();
    if tf.key.is_empty() {
        log::error!("could not process u64 filter");
        return false;
    }
    let tm = try_meta.unwrap_or_default();
    match tf.filter {
        FilterOperations::EqualTo => tf.eq(tm),
        FilterOperations::GreaterThan => tf.gt(tm),
        FilterOperations::GreaterThanEqualTo => tf.gte(tm),
        FilterOperations::LessThan => tf.lt(tm),
        FilterOperations::LessThanEqualTo => tf.lte(tm),
        _ => false,
    }
}

/// Proces two raw json strings. Let `raw_f` be a valid metadata filter
///
/// and `raw_m` be valid metadata that is not a nested object. Returns true
///
/// on a valid match. The equivalent of an SQL `where` clause.
pub fn filter_where(raw_f: &[String], raw_m: &[String]) -> bool {
    let mut t_count: usize = 0;
    let length = raw_f.len();
    for m in raw_m {
        for filter in raw_f {
            let tsf_result = process_string_filter(filter, m);
            if tsf_result {
                t_count += 1;
                if t_count == length {
                    return tsf_result;
                };
            }
            let tuf_result = process_u64_filter(filter, m);
            if tuf_result {
                t_count += 1;
                if t_count == length {
                    return tuf_result;
                };
            }
            let taf_result = process_string_array_filter(filter, m);
            if taf_result {
                t_count += 1;
                if t_count == length {
                    return taf_result;
                };
            }
        }
    }
    false // filters failed
}
