use log::{error, info};
use serde_json::{Error, Value};

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
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<String>, Error>;
    fn eq(self, m: MetadataFilter<String>) -> bool;
}

impl FilterString for MetadataFilter<String> {
    /// Create a filter on a valid string value
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<String>, Error> {
        for (index, jstr) in raw.iter().enumerate() {
            let v: Result<Value, serde_json::Error> = serde_json::from_str(jstr);
        if v.is_err() {
            error!("invalid json string");
            return Ok(Default::default());
        }
        let u_v: Value = v.unwrap();
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("could not parse string");
            return Ok(Default::default());
        }
        let mut key = vo.unwrap().keys().collect::<Vec<&String>>();
        key.sort();
        let key = key[index].to_string();
        let vo2 = vo.unwrap()[&key].as_object();
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_value = &u_v[&key];
            if !p_value.is_string() {
                return Ok(Default::default());
            }
            let value: String = p_value.as_str().unwrap().to_string();
            let filter: FilterOperations = FilterOperations::Noop;
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = vo2.unwrap().keys().collect::<Vec<&String>>()[0].to_string();
        let value = &vo2.unwrap()[&op];
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_string() {
            let value = value.as_str().unwrap().to_string();
            return Ok(MetadataFilter { key, filter, value });
        }
        }
        Ok(Default::default())
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
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<Vec<String>>, Error>;
    fn v_in(self, m: MetadataFilter<String>) -> bool;
}

impl FilterStringArray for MetadataFilter<Vec<String>> {
    /// Create a filter on a valid string value
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<Vec<String>>, Error> {
        for (index, jstr) in raw.iter().enumerate(){
            let v: Result<Value, serde_json::Error> = serde_json::from_str(jstr);
        if v.is_err() {
            error!("invalid json string");
            return Ok(Default::default());
        }
        let u_v: Value = v.unwrap();
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("failed to parse string array");
            return Ok(Default::default());
        }
        let mut key = vo.unwrap().keys().collect::<Vec<&String>>();
        key.sort();
        let key = key[index].to_string();
        let vo2 = vo.unwrap()[&key].as_object();
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_array = &u_v[&key];
            if !p_array.is_array() {
                return Ok(Default::default());
            }
            let u_array = p_array.as_array().unwrap();
            let filter: FilterOperations = FilterOperations::Noop;
            // duck invalid values in the array and fail the metadata filter
            let value: Vec<String> = u_array
                .iter()
                .map(|s| String::from(s.as_str().unwrap_or_default()))
                .collect();
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = vo2.unwrap().keys().collect::<Vec<&String>>()[0].to_string();
        let value = &vo2.unwrap()[&op];
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_array() {
            let possible_array = value.as_array().unwrap();
            // duck invalid values in the array and fail the metadata filter
            let value = possible_array
                .iter()
                .map(|s| String::from(s.as_str().unwrap_or_default()))
                .collect();
            return Ok(MetadataFilter { key, filter, value });
        }
        }
        Ok(Default::default())
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
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<u64>, Error>;
    fn eq(self, m: MetadataFilter<u64>) -> bool;
    fn gt(self, m: MetadataFilter<u64>) -> bool;
    fn gte(self, m: MetadataFilter<u64>) -> bool;
    fn lt(self, m: MetadataFilter<u64>) -> bool;
    fn lte(self, m: MetadataFilter<u64>) -> bool;
}

impl Filteru64 for MetadataFilter<u64> {
    /// Create a filter on a valid u64 value
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<u64>, Error> {
        for (index, jstr) in raw.iter().enumerate() {
            let v: Result<Value, serde_json::Error> = serde_json::from_str(jstr);
        if v.is_err() {
            error!("invalid json string");
            return Ok(Default::default());
        }
        let u_v: Value = v.unwrap();
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("failed to parse u64");
            return Ok(Default::default());
        }
        let mut key = vo.unwrap().keys().collect::<Vec<&String>>();
        key.sort();
        let key = key[index].to_string();
        log::debug!("u64 key={}", key);
        let vo2 = vo.unwrap()[&key].as_object();
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_value = &u_v[&key];
            if !p_value.is_u64() {
                return Ok(Default::default());
            }
            let value: u64 = p_value.as_u64().unwrap();
            let filter: FilterOperations = FilterOperations::Noop;
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = vo2.unwrap().keys().collect::<Vec<&String>>()[0].to_string();
        let value = &vo2.unwrap()[&op];
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_u64() {
            let value = value.as_u64().unwrap();
            return Ok(MetadataFilter { key, filter, value });
        }
        }
        Ok(Default::default())
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

impl Default for MetadataFilter<i64> {
    fn default() -> Self {
        MetadataFilter {
            key: Default::default(),
            value: 0,
            filter: FilterOperations::Noop,
        }
    }
}

trait Filteri64 {
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<i64>, Error>;
    fn eq(self, m: MetadataFilter<i64>) -> bool;
    fn gt(self, m: MetadataFilter<i64>) -> bool;
    fn gte(self, m: MetadataFilter<i64>) -> bool;
    fn lt(self, m: MetadataFilter<i64>) -> bool;
    fn lte(self, m: MetadataFilter<i64>) -> bool;
}

impl Filteri64 for MetadataFilter<i64> {
    /// Create a filter on a valid i64 value
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<i64>, Error> {
        for (index, jstr) in raw.iter().enumerate() {
            let v: Result<Value, serde_json::Error> = serde_json::from_str(jstr);
        if v.is_err() {
            error!("invalid json string");
            return Ok(Default::default());
        }
        let u_v: Value = v.unwrap();
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("failed to parse i64");
            return Ok(Default::default());
        }
        let mut key = vo.unwrap().keys().collect::<Vec<&String>>();
        key.sort();
        let key = key[index].to_string();
        let vo2 = vo.unwrap()[&key].as_object();
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_value = &u_v[&key];
            if !p_value.is_i64() {
                return Ok(Default::default());
            }
            let value: i64 = p_value.as_i64().unwrap();
            let filter: FilterOperations = FilterOperations::Noop;
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = vo2.unwrap().keys().collect::<Vec<&String>>()[0].to_string();
        let value = &vo2.unwrap()[&op];
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_i64() {
            let value = value.as_i64().unwrap();
            return Ok(MetadataFilter { key, filter, value });
        }
        }
        Ok(Default::default())
    }
    fn eq(self, m: MetadataFilter<i64>) -> bool {
        match self.filter {
            FilterOperations::EqualTo => self.key == m.key && self.value == m.value,
            _ => false,
        }
    }
    fn gt(self, m: MetadataFilter<i64>) -> bool {
        match self.filter {
            FilterOperations::GreaterThan => self.key == m.key && self.value < m.value,
            _ => false,
        }
    }
    fn gte(self, m: MetadataFilter<i64>) -> bool {
        match self.filter {
            FilterOperations::GreaterThanEqualTo => self.key == m.key && self.value <= m.value,
            _ => false,
        }
    }
    fn lt(self, m: MetadataFilter<i64>) -> bool {
        match self.filter {
            FilterOperations::LessThan => self.key == m.key && self.value > m.value,
            _ => false,
        }
    }
    fn lte(self, m: MetadataFilter<i64>) -> bool {
        match self.filter {
            FilterOperations::LessThanEqualTo => self.key == m.key && self.value >= m.value,
            _ => false,
        }
    }
}

impl Default for MetadataFilter<f64> {
    fn default() -> Self {
        MetadataFilter {
            key: Default::default(),
            value: 0.0,
            filter: FilterOperations::Noop,
        }
    }
}

trait Filterf64 {
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<f64>, Error>;
    fn eq(self, m: MetadataFilter<f64>) -> bool;
    fn gt(self, m: MetadataFilter<f64>) -> bool;
    fn gte(self, m: MetadataFilter<f64>) -> bool;
    fn lt(self, m: MetadataFilter<f64>) -> bool;
    fn lte(self, m: MetadataFilter<f64>) -> bool;
}

impl Filterf64 for MetadataFilter<f64> {
    /// Create a filter on a valid f64 value
    fn create_filter(raw: Vec<String>) -> Result<MetadataFilter<f64>, Error> {
        for (index, jstr) in raw.iter().enumerate() {
            let v: Result<Value, serde_json::Error> = serde_json::from_str(jstr);
        if v.is_err() {
            error!("invalid json string");
            return Ok(Default::default());
        }
        let u_v: Value = v.unwrap();
        let vo = u_v.as_object();
        if vo.is_none() {
            error!("failed to parse f64");
            return Ok(Default::default());
        }
        let mut key = vo.unwrap().keys().collect::<Vec<&String>>();
        key.sort();
        let key = key[index].to_string();
        let vo2 = vo.unwrap()[&key].as_object();
        if vo2.is_none() {
            info!("no op key found, processing as metadata");
            let p_value = &u_v[&key];
            if !p_value.is_f64() {
                return Ok(Default::default());
            }
            let value: f64 = p_value.as_f64().unwrap();
            let filter: FilterOperations = FilterOperations::Noop;
            return Ok(MetadataFilter { key, filter, value });
        }
        let op = vo2.unwrap().keys().collect::<Vec<&String>>()[0].to_string();
        let value = &vo2.unwrap()[&op];
        let filter: FilterOperations = FilterOperations::get_enum(&op);
        if value.is_f64() {
            let value = value.as_f64().unwrap();
            return Ok(MetadataFilter { key, filter, value });
        }
        }
        Ok(Default::default())
    }
    fn eq(self, m: MetadataFilter<f64>) -> bool {
        match self.filter {
            FilterOperations::EqualTo => self.key == m.key && self.value == m.value,
            _ => false,
        }
    }
    fn gt(self, m: MetadataFilter<f64>) -> bool {
        match self.filter {
            FilterOperations::GreaterThan => self.key == m.key && self.value < m.value,
            _ => false,
        }
    }
    fn gte(self, m: MetadataFilter<f64>) -> bool {
        match self.filter {
            FilterOperations::GreaterThanEqualTo => self.key == m.key && self.value <= m.value,
            _ => false,
        }
    }
    fn lt(self, m: MetadataFilter<f64>) -> bool {
        match self.filter {
            FilterOperations::LessThan => self.key == m.key && self.value > m.value,
            _ => false,
        }
    }
    fn lte(self, m: MetadataFilter<f64>) -> bool {
        match self.filter {
            FilterOperations::LessThanEqualTo => self.key == m.key && self.value >= m.value,
            _ => false,
        }
    }
}

fn process_string_filter(raw_f: Vec<String>, raw_m: Vec<String>) -> bool {
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

fn process_string_array_filter(raw_f: Vec<String>, raw_m: Vec<String>) -> bool {
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

fn process_u64_filter(raw_f: Vec<String>, raw_m: Vec<String>) -> bool {
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

fn process_i64_filter(raw_f: Vec<String>, raw_m: Vec<String>) -> bool {
    let try_filter = <MetadataFilter<i64> as Filteri64>::create_filter(raw_f);
    let try_meta = <MetadataFilter<i64> as Filteri64>::create_filter(raw_m);
    let tf = try_filter.unwrap_or_default();
    if tf.key.is_empty() {
        log::error!("could not process i64 filter");
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

fn process_f64_filter(raw_f: Vec<String>, raw_m: Vec<String>) -> bool {
    let try_filter = <MetadataFilter<f64> as Filterf64>::create_filter(raw_f);
    let try_meta = <MetadataFilter<f64> as Filterf64>::create_filter(raw_m);
    let tf = try_filter.unwrap_or_default();
    if tf.key.is_empty() {
        log::error!("could not process i64 filter");
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
pub fn filter_where(raw_f: &Vec<String>, raw_m: &[String]) -> bool {
    let length = raw_f.len();
    for filter in raw_f {
        let mut truthy_counter: usize = 0;
        let tsf_result = process_string_filter(vec![filter.to_string()], raw_m.to_vec());
        if tsf_result { truthy_counter += 1; }
        if tsf_result && truthy_counter == length {
            return tsf_result;
        }
        let tuf_result = process_u64_filter(vec![filter.to_string()], raw_m.to_vec());
        log::debug!("u64 filter input {} and {:?}", filter.to_string(), raw_m.to_vec());
        if tuf_result { truthy_counter += 1; }
        if tuf_result && truthy_counter == length {
            return tuf_result;
        }
        let tif_result = process_i64_filter(vec![filter.to_string()], raw_m.to_vec());
        if tif_result && truthy_counter == length {
            return tif_result;
        }
        let tff_result = process_f64_filter(vec![filter.to_string()], raw_m.to_vec());
        if tff_result && truthy_counter == length {
            return tff_result;
        }
        let taf_result = process_string_array_filter(vec![filter.to_string()], raw_m.to_vec());
        if taf_result && truthy_counter == length {
            return taf_result;
        }
    }
    false // filters failed
}
