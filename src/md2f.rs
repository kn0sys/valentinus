use log::{debug, info};
use serde_json::Value;

/// Possible errors while filtering may be due to
///
/// parsing, invalid operation value etc.
#[derive(Debug)]
pub enum Md2fsError {
    SerdeJsonError,
    ParseError,
}
/// Where clause keys
#[derive(Debug)]
enum FilterOperations {
    EqualTo,
    GreaterThanEqualTo,
    GreaterThan,
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
            "lt" => FilterOperations::LessThan,
            "lte" => FilterOperations::LessThanEqualTo,
            _ => FilterOperations::EqualTo,
        }
    }
}

#[derive(Debug)]
enum MetadataFilterResult {
    U64Filter(MetadataFilter<u64>),
    StringFilter(MetadataFilter<String>),
}

/// Metadata filter
#[derive(Debug)]
struct MetadataFilter<T> {
    /// Key to filter on
    key: String,
    /// Valid json type to filter on
    value: T,
    /// Filter operations eq, gt, gte, in, lt, lte
    filter: FilterOperations,
}

impl<T: Default> Default for MetadataFilter<T> {
    fn default() -> Self {
        MetadataFilter {
            key: Default::default(),
            value: Default::default(),
            filter: FilterOperations::Noop,
        }
    }
}

trait Filter<T> {
    fn create_filter(raw: &str) -> Result<MetadataFilterResult, Md2fsError>;
    fn eq(self, m: MetadataFilter<T>) -> bool;
    fn gt(self, m: MetadataFilter<T>) -> bool;
    fn gte(self, m: MetadataFilter<T>) -> bool;
    fn lt(self, m: MetadataFilter<T>) -> bool;
    fn lte(self, m: MetadataFilter<T>) -> bool;
}

impl<T> Filter<T> for MetadataFilter<T>
where
    T: PartialEq + PartialOrd + Default,
{
    /// Create a filter on a valid string value
    fn create_filter(raw: &str) -> Result<MetadataFilterResult, Md2fsError> {
        let v: Result<Value, serde_json::Error> = serde_json::from_str(raw);
        if v.is_err() {
            debug!("invalid json string");
            return Err(Md2fsError::SerdeJsonError);
        }
        let u_v: Value = v.map_err(|_| Md2fsError::ParseError)?;
        let vo = u_v.as_object();
        if vo.is_none() {
            debug!("could not parse string");
        }
        let key = match vo {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _ => String::new(),
        };
        let vo2 = match vo {
            Some(v) => v[&key].as_object(),
            _ => None,
        };
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_value = &u_v[&key];
            if !p_value.is_string() {
                debug!("op key is not a string value");
            }
            if p_value.is_string() {
                let value: String = match p_value.as_str() {
                    Some(s) => s.to_string(),
                    _ => String::new(),
                };
                let filter: FilterOperations = FilterOperations::Noop;
                return Ok(MetadataFilterResult::StringFilter(MetadataFilter {
                    key,
                    filter,
                    value,
                }));
            } else {
                let value: u64 = p_value.as_u64().unwrap_or_default();
                let filter: FilterOperations = FilterOperations::Noop;
                return Ok(MetadataFilterResult::U64Filter(MetadataFilter {
                    key,
                    filter,
                    value,
                }));
            }
        }
        let op = match vo2 {
            Some(v) => v.keys().collect::<Vec<&String>>()[0].to_string(),
            _ => String::new(),
        };
        let value = match vo2 {
            Some(v) => &v[&op],
            _ => &Value::String(String::new()),
        };
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_string() {
            let value = match value.as_str() {
                Some(s) => s.to_string(),
                _ => String::new(),
            };
            return Ok(MetadataFilterResult::StringFilter(MetadataFilter {
                key,
                filter,
                value,
            }));
        }
        if value.is_u64() {
            let value = value.as_u64().unwrap_or_default();
            return Ok(MetadataFilterResult::U64Filter(MetadataFilter {
                key,
                filter,
                value,
            }));
        }
        Ok(MetadataFilterResult::StringFilter(Default::default()))
    }
    fn eq(self, m: MetadataFilter<T>) -> bool {
        match self.filter {
            FilterOperations::EqualTo => self.key == m.key && self.value == m.value,
            _ => false,
        }
    }
    fn gt(self, m: MetadataFilter<T>) -> bool {
        match self.filter {
            FilterOperations::GreaterThan => self.key == m.key && self.value < m.value,
            _ => false,
        }
    }
    fn gte(self, m: MetadataFilter<T>) -> bool {
        match self.filter {
            FilterOperations::GreaterThanEqualTo => self.key == m.key && self.value <= m.value,
            _ => false,
        }
    }
    fn lt(self, m: MetadataFilter<T>) -> bool {
        match self.filter {
            FilterOperations::LessThan => self.key == m.key && self.value > m.value,
            _ => false,
        }
    }
    fn lte(self, m: MetadataFilter<T>) -> bool {
        match self.filter {
            FilterOperations::LessThanEqualTo => self.key == m.key && self.value >= m.value,
            _ => false,
        }
    }
}

fn process_filter(raw_f: &str, raw_m: &str) -> Result<bool, Md2fsError> {
    let try_filter: Result<MetadataFilterResult, Md2fsError> =
        MetadataFilter::<String>::create_filter(raw_f);
    let try_meta: Result<MetadataFilterResult, Md2fsError> =
        MetadataFilter::<String>::create_filter(raw_m);
    debug!("debug -> {:?}", try_meta);
    let tf = try_filter?;
    let tm = try_meta?;
    let string_filter = match tf {
        MetadataFilterResult::StringFilter(fstr) => match tm {
            MetadataFilterResult::StringFilter(mstr) => match fstr.filter {
                FilterOperations::EqualTo => Ok(fstr.eq(mstr)),
                _ => Ok(false),
            },
            _ => Ok(false),
        },
        _ => Ok(false),
    }?;
    let try_filter: Result<MetadataFilterResult, Md2fsError> =
        MetadataFilter::<u64>::create_filter(raw_f);
    let try_meta: Result<MetadataFilterResult, Md2fsError> =
        MetadataFilter::<u64>::create_filter(raw_m);
    let tf = try_filter?;
    let tm = try_meta?;
    let u64_filter = match tf {
        MetadataFilterResult::U64Filter(fu64) => match tm {
            MetadataFilterResult::U64Filter(mu64) => match fu64.filter {
                FilterOperations::EqualTo => Ok(fu64.eq(mu64)),
                FilterOperations::GreaterThan => Ok(fu64.gt(mu64)),
                FilterOperations::GreaterThanEqualTo => Ok(fu64.gte(mu64)),
                FilterOperations::LessThan => Ok(fu64.lt(mu64)),
                FilterOperations::LessThanEqualTo => Ok(fu64.lte(mu64)),
                _ => Ok(false),
            },
            _ => Ok(false),
        },
        _ => Ok(false),
    }?;
    Ok(string_filter || u64_filter)
}

/// Proces two raw json strings. Let `raw_f` be a valid metadata filter
///
/// and `raw_m` be valid metadata that is not a nested object. Returns true
///
/// on a valid match. The equivalent of an SQL `where` clause.
pub fn filter_where(raw_f: &[String], raw_m: &[String]) -> Result<bool, Md2fsError> {
    let mut t_count: usize = 0;
    let length = raw_f.len();
    for m in raw_m {
        for filter in raw_f {
            let tsf_result = process_filter(filter, m)?;
            if tsf_result {
                t_count += 1;
                if t_count == length {
                    return Ok(tsf_result);
                };
            }
        }
    }
    Ok(false) // filters failed
}
