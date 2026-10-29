//! Credentials that are not keys in the environment (GPU-309).
//!
//! A company that runs training on AWS gives the machine a role, not a key:
//! an instance profile on EC2, a task role on ECS, a service account mapped
//! to a role on EKS. Before this module Moonclip read keys only from the
//! environment, so on such a machine it could not reach the bucket at all,
//! and the way around it was to mint a long-lived key and put it where the
//! role was meant to make keys unnecessary.
//!
//! Looked for in the order the AWS SDKs use, after the environment's keys
//! ([`crate::s3::S3Credentials`]) and before giving up:
//!
//! 1. **A profile** in the shared credentials file: `AWS_PROFILE` (else
//!    `default`) in `AWS_SHARED_CREDENTIALS_FILE` (else `~/.aws/credentials`).
//!    Static keys, read once. Only the keys of that file: a profile in
//!    `~/.aws/config` that assumes a role or signs in with SSO is not
//!    followed.
//! 2. **Web identity** (EKS's IRSA, and any OIDC federation):
//!    `AWS_WEB_IDENTITY_TOKEN_FILE` and `AWS_ROLE_ARN`, exchanged at STS with
//!    `AssumeRoleWithWebIdentity`. That call is not signed: the token is the
//!    proof.
//! 3. **The container's credentials** (ECS, and EKS Pod Identity):
//!    `AWS_CONTAINER_CREDENTIALS_FULL_URI` or `..._RELATIVE_URI`, with the
//!    authorization token those set beside them.
//! 4. **The instance's role**, from the EC2 metadata service, IMDSv2 only:
//!    a session token first, then the role's credentials. Not asked when
//!    `AWS_EC2_METADATA_DISABLED=true`, and given one second to answer, so a
//!    machine that is not on EC2 fails as it did before, a second later.
//!
//! The last three hand out credentials that expire, within hours. They are
//! fetched again five minutes before that, on the request that finds them
//! due, so a run longer than the credentials' life keeps writing: what the
//! old documentation called the caller's problem ("one that expires mid-run
//! fails like a wrong key") is now this module's.

use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use crate::error::{MoonclipError, Result};
use crate::s3::S3Credentials;

/// Fetched again this long before they expire: long enough that a request
/// signed just before the refresh is not refused mid-flight, short against
/// the hour-plus that every one of these sources hands out.
const REFRESH_BEFORE: Duration = Duration::from_secs(300);

/// How long a metadata service has to answer. One second, like the SDKs': it
/// is a link-local address that answers in milliseconds where it exists.
const METADATA_TIMEOUT: Duration = Duration::from_secs(1);

/// Where the credentials of a role come from. See the module's list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoleSource {
    WebIdentity { token_file: String, role_arn: String, session_name: String, sts: String },
    Container { url: String, authorization: Option<Authorization> },
    Instance { endpoint: String },
}

/// The token a container's credentials endpoint wants, as given or as a
/// file to read on every fetch (EKS Pod Identity rotates it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Authorization {
    Value(String),
    File(String),
}

/// A role's credentials, kept until they are due again.
pub struct RoleCredentials {
    source: RoleSource,
    agent: ureq::Agent,
    current: Mutex<Option<(S3Credentials, Option<SystemTime>)>>,
}

impl std::fmt::Debug for RoleCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The source, never the keys.
        f.debug_struct("RoleCredentials").field("source", &self.source).finish()
    }
}

impl RoleCredentials {
    pub fn new(source: RoleSource) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(METADATA_TIMEOUT)
            .timeout(Duration::from_secs(10))
            .build();
        RoleCredentials { source, agent, current: Mutex::new(None) }
    }

    pub fn source(&self) -> &RoleSource {
        &self.source
    }

    /// The credentials to sign with now, fetched when there are none yet or
    /// they expire within [`REFRESH_BEFORE`].
    pub fn get(&self) -> Result<S3Credentials> {
        let mut current = self.current.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((keys, expires)) = current.as_ref() {
            let due = expires.is_some_and(|at| {
                at.duration_since(SystemTime::now()).map_or(true, |left| left < REFRESH_BEFORE)
            });
            if !due {
                return Ok(keys.clone());
            }
        }
        let fetched = self.fetch()?;
        *current = Some(fetched.clone());
        Ok(fetched.0)
    }

    fn fetch(&self) -> Result<(S3Credentials, Option<SystemTime>)> {
        match &self.source {
            RoleSource::WebIdentity { token_file, role_arn, session_name, sts } => {
                let token = std::fs::read_to_string(token_file).map_err(|e| {
                    MoonclipError::Config(format!(
                        "AWS_WEB_IDENTITY_TOKEN_FILE {token_file:?} could not be read: {e}"
                    ))
                })?;
                let response = self
                    .agent
                    .get(sts)
                    .query("Action", "AssumeRoleWithWebIdentity")
                    .query("Version", "2011-06-15")
                    .query("RoleArn", role_arn)
                    .query("RoleSessionName", session_name)
                    .query("WebIdentityToken", token.trim())
                    .call()
                    .map_err(|e| failed("STS AssumeRoleWithWebIdentity", e))?;
                let body = response
                    .into_string()
                    .map_err(|e| MoonclipError::Storage(format!("STS answer: {e}")))?;
                let value = |name: &str| crate::s3::extract_xml_value(&body, name);
                match (value("AccessKeyId"), value("SecretAccessKey"), value("SessionToken")) {
                    (Some(access_key), Some(secret_key), Some(token)) => Ok((
                        S3Credentials { access_key, secret_key, session_token: Some(token) },
                        value("Expiration").and_then(|at| expiration(&at)),
                    )),
                    _ => Err(MoonclipError::Storage(
                        "STS answered AssumeRoleWithWebIdentity without credentials".into(),
                    )),
                }
            }
            RoleSource::Container { url, authorization } => {
                let mut request = self.agent.get(url);
                if let Some(authorization) = authorization {
                    let value = match authorization {
                        Authorization::Value(value) => value.clone(),
                        Authorization::File(path) => std::fs::read_to_string(path)
                            .map_err(|e| {
                                MoonclipError::Config(format!(
                                    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE {path:?} could not be read: {e}"
                                ))
                            })?
                            .trim()
                            .to_string(),
                    };
                    request = request.set("Authorization", &value);
                }
                let body = request
                    .call()
                    .map_err(|e| failed("the container's credentials endpoint", e))?
                    .into_string()
                    .map_err(|e| MoonclipError::Storage(format!("container credentials: {e}")))?;
                from_json(&body, "the container's credentials endpoint")
            }
            RoleSource::Instance { endpoint } => {
                let base = endpoint.trim_end_matches('/');
                let session = self
                    .agent
                    .put(&format!("{base}/latest/api/token"))
                    .set("X-aws-ec2-metadata-token-ttl-seconds", "21600")
                    .call()
                    .map_err(|e| failed("the instance metadata service (IMDSv2 token)", e))?
                    .into_string()
                    .map_err(|e| MoonclipError::Storage(format!("IMDSv2 token: {e}")))?;
                let read = |path: &str| -> Result<String> {
                    self.agent
                        .get(&format!("{base}{path}"))
                        .set("X-aws-ec2-metadata-token", session.trim())
                        .call()
                        .map_err(|e| failed("the instance metadata service", e))?
                        .into_string()
                        .map_err(|e| MoonclipError::Storage(format!("instance metadata: {e}")))
                };
                let roles = read("/latest/meta-data/iam/security-credentials/")?;
                let role = roles.lines().map(str::trim).find(|line| !line.is_empty()).ok_or_else(|| {
                    MoonclipError::Config(
                        "this instance has no IAM role: attach an instance profile, or give keys".into(),
                    )
                })?;
                let body = read(&format!("/latest/meta-data/iam/security-credentials/{role}"))?;
                from_json(&body, "the instance metadata service")
            }
        }
    }
}

/// The role this machine has, from what the environment says, or None. A
/// profile's static keys are not a role: see [`profile_keys`].
///
/// `var` is the environment, as a function so a test can give one.
pub fn detect(var: impl Fn(&str) -> Option<String>) -> Option<RoleSource> {
    let read = |name: &str| var(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    if let (Some(token_file), Some(role_arn)) = (read("AWS_WEB_IDENTITY_TOKEN_FILE"), read("AWS_ROLE_ARN")) {
        let sts = read("AWS_ENDPOINT_URL_STS").unwrap_or_else(|| {
            match read("AWS_REGION").or_else(|| read("AWS_DEFAULT_REGION")) {
                Some(region) => format!("https://sts.{region}.amazonaws.com/"),
                None => "https://sts.amazonaws.com/".into(),
            }
        });
        let session_name = read("AWS_ROLE_SESSION_NAME").unwrap_or_else(|| "moonclip".into());
        return Some(RoleSource::WebIdentity { token_file, role_arn, session_name, sts });
    }
    let container = read("AWS_CONTAINER_CREDENTIALS_FULL_URI").or_else(|| {
        read("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").map(|path| format!("http://169.254.170.2{path}"))
    });
    if let Some(url) = container {
        let authorization = read("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")
            .map(Authorization::File)
            .or_else(|| read("AWS_CONTAINER_AUTHORIZATION_TOKEN").map(Authorization::Value));
        return Some(RoleSource::Container { url, authorization });
    }
    if read("AWS_EC2_METADATA_DISABLED").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
        return None;
    }
    let endpoint =
        read("AWS_EC2_METADATA_SERVICE_ENDPOINT").unwrap_or_else(|| "http://169.254.169.254".into());
    Some(RoleSource::Instance { endpoint })
}

/// The keys of a profile in the shared credentials file, or None when there
/// is no file, no such profile, or a profile without both keys.
///
/// The file is INI as the AWS CLI writes it: `[name]` sections of
/// `key = value`. `AWS_PROFILE` names the section, `default` without it.
pub fn profile_keys(var: impl Fn(&str) -> Option<String>) -> Option<S3Credentials> {
    let read = |name: &str| var(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let path = read("AWS_SHARED_CREDENTIALS_FILE")
        .or_else(|| read("HOME").or_else(|| read("USERPROFILE")).map(|home| format!("{home}/.aws/credentials")))?;
    let text = std::fs::read_to_string(path).ok()?;
    let wanted = read("AWS_PROFILE").unwrap_or_else(|| "default".into());
    parse_profile(&text, &wanted)
}

fn parse_profile(text: &str, wanted: &str) -> Option<S3Credentials> {
    let mut inside = false;
    let (mut access_key, mut secret_key, mut session_token) = (None, None, None);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            inside = name.trim() == wanted;
            continue;
        }
        if !inside {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let value = Some(value.trim().to_string()).filter(|v| !v.is_empty());
            match key.trim().to_ascii_lowercase().as_str() {
                "aws_access_key_id" => access_key = value,
                "aws_secret_access_key" => secret_key = value,
                "aws_session_token" => session_token = value,
                _ => {}
            }
        }
    }
    Some(S3Credentials { access_key: access_key?, secret_key: secret_key?, session_token })
}

/// The JSON a container endpoint and the instance metadata service answer
/// with: `AccessKeyId`, `SecretAccessKey`, `Token`, `Expiration`.
fn from_json(body: &str, from: &str) -> Result<(S3Credentials, Option<SystemTime>)> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| MoonclipError::Storage(format!("{from} answered something that is not JSON: {e}")))?;
    let text = |name: &str| value.get(name).and_then(|v| v.as_str()).map(str::to_string);
    match (text("AccessKeyId"), text("SecretAccessKey")) {
        (Some(access_key), Some(secret_key)) => Ok((
            S3Credentials { access_key, secret_key, session_token: text("Token") },
            text("Expiration").and_then(|at| expiration(&at)),
        )),
        _ => Err(MoonclipError::Storage(format!("{from} answered without credentials"))),
    }
}

/// An ISO 8601 time, as all three sources write `Expiration`.
fn expiration(at: &str) -> Option<SystemTime> {
    let at: jiff::Timestamp = at.trim().parse().ok()?;
    SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(u64::try_from(at.as_second()).ok()?))
}

fn failed(what: &str, e: ureq::Error) -> MoonclipError {
    match e {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            let body = body.trim();
            if body.is_empty() {
                MoonclipError::Storage(format!("{what} answered {code}"))
            } else {
                MoonclipError::Storage(format!("{what} answered {code}: {body}"))
            }
        }
        ureq::Error::Transport(t) => MoonclipError::Storage(format!("{what} did not answer: {t}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name| map.get(name).cloned()
    }

    /// A server answering each request with `answer(method, path, headers)`,
    /// and the requests it saw.
    fn serve(
        answer: impl Fn(&str, &str, &HashMap<String, String>) -> (u16, String) + Send + Sync + 'static,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut headers = HashMap::new();
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = line.split_once(':') {
                        let k = k.trim().to_ascii_lowercase();
                        if k == "content-length" {
                            length = v.trim().parse().unwrap_or(0);
                        }
                        headers.insert(k, v.trim().to_string());
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let mut parts = first.split_whitespace();
                let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                log.lock().unwrap().push(format!("{method} {path}"));
                let (code, body) = answer(method, path, &headers);
                write!(
                    stream,
                    "HTTP/1.1 {code} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
        });
        (address, seen)
    }

    #[test]
    fn a_profile_is_read_from_the_shared_file() {
        let text = "[default]\naws_access_key_id = AKD\naws_secret_access_key = SKD\n\n\
                    [work]\n# a comment\naws_access_key_id=AKW\naws_secret_access_key=SKW\naws_session_token = T\n";
        assert_eq!(
            parse_profile(text, "default"),
            Some(S3Credentials { access_key: "AKD".into(), secret_key: "SKD".into(), session_token: None })
        );
        assert_eq!(
            parse_profile(text, "work"),
            Some(S3Credentials { access_key: "AKW".into(), secret_key: "SKW".into(), session_token: Some("T".into()) })
        );
        assert_eq!(parse_profile(text, "other"), None);
        assert_eq!(parse_profile("[default]\naws_access_key_id = only\n", "default"), None);
    }

    #[test]
    fn aws_profile_picks_the_section_of_the_named_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("credentials");
        std::fs::write(&file, "[default]\naws_access_key_id=A\naws_secret_access_key=B\n[ci]\naws_access_key_id=C\naws_secret_access_key=D\n").unwrap();
        let path = file.to_str().unwrap();
        let found = profile_keys(env(&[("AWS_SHARED_CREDENTIALS_FILE", path), ("AWS_PROFILE", "ci")])).unwrap();
        assert_eq!(found.access_key, "C");
        let found = profile_keys(env(&[("AWS_SHARED_CREDENTIALS_FILE", path)])).unwrap();
        assert_eq!(found.access_key, "A");
        assert!(profile_keys(env(&[("HOME", dir.path().to_str().unwrap())])).is_none());
    }

    #[test]
    fn the_sources_are_found_in_the_sdks_order() {
        let both = env(&[
            ("AWS_WEB_IDENTITY_TOKEN_FILE", "/t"),
            ("AWS_ROLE_ARN", "arn:aws:iam::1:role/r"),
            ("AWS_REGION", "eu-west-1"),
            ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/x"),
        ]);
        assert_eq!(
            detect(both),
            Some(RoleSource::WebIdentity {
                token_file: "/t".into(),
                role_arn: "arn:aws:iam::1:role/r".into(),
                session_name: "moonclip".into(),
                sts: "https://sts.eu-west-1.amazonaws.com/".into(),
            })
        );
        assert_eq!(
            detect(env(&[("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/x")])),
            Some(RoleSource::Container { url: "http://169.254.170.2/v2/credentials/x".into(), authorization: None })
        );
        assert_eq!(detect(env(&[])), Some(RoleSource::Instance { endpoint: "http://169.254.169.254".into() }));
        assert_eq!(detect(env(&[("AWS_EC2_METADATA_DISABLED", "true")])), None);
    }

    #[test]
    fn the_instance_role_is_read_with_an_imdsv2_session() {
        let (address, seen) = serve(|method, path, headers| match (method, path) {
            ("PUT", "/latest/api/token") => {
                assert_eq!(headers.get("x-aws-ec2-metadata-token-ttl-seconds").map(String::as_str), Some("21600"));
                (200, "SESSION".into())
            }
            ("GET", _) if headers.get("x-aws-ec2-metadata-token").map(String::as_str) != Some("SESSION") => {
                (401, String::new())
            }
            ("GET", "/latest/meta-data/iam/security-credentials/") => (200, "trainer\n".into()),
            ("GET", "/latest/meta-data/iam/security-credentials/trainer") => (
                200,
                r#"{"Code":"Success","AccessKeyId":"ASIA1","SecretAccessKey":"S1","Token":"T1","Expiration":"2099-01-01T00:00:00Z"}"#.into(),
            ),
            _ => (404, String::new()),
        });
        let role = RoleCredentials::new(RoleSource::Instance { endpoint: address });
        let keys = role.get().unwrap();
        assert_eq!(keys, S3Credentials { access_key: "ASIA1".into(), secret_key: "S1".into(), session_token: Some("T1".into()) });
        // Kept: far from expiring, so no second round of requests.
        role.get().unwrap();
        assert_eq!(seen.lock().unwrap().len(), 3);
    }

    #[test]
    fn credentials_about_to_expire_are_fetched_again() {
        let (address, seen) = serve(|_, _, _| {
            // Expiring in a minute: inside the refresh window.
            let soon = jiff::Timestamp::now() + jiff::SignedDuration::from_secs(60);
            (200, format!(r#"{{"AccessKeyId":"A","SecretAccessKey":"S","Token":"T","Expiration":"{soon}"}}"#))
        });
        let role = RoleCredentials::new(RoleSource::Container { url: format!("{address}/creds"), authorization: None });
        role.get().unwrap();
        role.get().unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[test]
    fn the_container_endpoint_gets_its_authorization_token() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "secret-token\n").unwrap();
        let (address, _) = serve(|_, path, headers| {
            assert_eq!(path, "/v2/credentials");
            match headers.get("authorization").map(String::as_str) {
                Some("secret-token") => (200, r#"{"AccessKeyId":"A","SecretAccessKey":"S","Token":"T"}"#.into()),
                _ => (403, String::new()),
            }
        });
        let role = RoleCredentials::new(RoleSource::Container {
            url: format!("{address}/v2/credentials"),
            authorization: Some(Authorization::File(token.to_str().unwrap().into())),
        });
        assert_eq!(role.get().unwrap().access_key, "A");
    }

    #[test]
    fn a_web_identity_token_is_exchanged_at_sts() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "eyJ.oidc.token\n").unwrap();
        let (address, seen) = serve(|method, path, _| {
            assert_eq!(method, "GET");
            assert!(path.contains("Action=AssumeRoleWithWebIdentity"), "{path}");
            assert!(path.contains("WebIdentityToken=eyJ.oidc.token"), "{path}");
            (
                200,
                "<AssumeRoleWithWebIdentityResponse><AssumeRoleWithWebIdentityResult><Credentials>\
                 <AccessKeyId>ASIAW</AccessKeyId><SecretAccessKey>SW</SecretAccessKey>\
                 <SessionToken>TW</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration>\
                 </Credentials></AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>"
                    .into(),
            )
        });
        let role = RoleCredentials::new(RoleSource::WebIdentity {
            token_file: token.to_str().unwrap().into(),
            role_arn: "arn:aws:iam::1:role/r".into(),
            session_name: "moonclip".into(),
            sts: format!("{address}/"),
        });
        assert_eq!(
            role.get().unwrap(),
            S3Credentials { access_key: "ASIAW".into(), secret_key: "SW".into(), session_token: Some("TW".into()) }
        );
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn keys_come_first_then_a_profile_then_a_role() {
        use crate::s3::{signing_with, Signing};
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().to_str().unwrap().to_string();
        std::fs::create_dir(dir.path().join(".aws")).unwrap();
        std::fs::write(dir.path().join(".aws/credentials"), "[default]\naws_access_key_id=P\naws_secret_access_key=Q\n").unwrap();

        let keys = |s: Signing| match s {
            Signing::Keys(k) => k.access_key,
            Signing::Role(_) => "role".into(),
        };
        // The environment's keys win over the profile.
        let found = signing_with(None, None, None, env(&[("HOME", &home), ("AWS_ACCESS_KEY_ID", "E"), ("AWS_SECRET_ACCESS_KEY", "F")]));
        assert_eq!(keys(found.unwrap()), "E");
        // Without them, the profile.
        assert_eq!(keys(signing_with(None, None, None, env(&[("HOME", &home)])).unwrap()), "P");
        // Keys given are never completed or replaced from elsewhere.
        assert!(signing_with(Some("only"), None, None, env(&[("HOME", &home)])).is_err());

        // Nothing at all, with the metadata service off: the old error.
        let nothing = signing_with(None, None, None, env(&[("AWS_EC2_METADATA_DISABLED", "true")]));
        let error = nothing.err().unwrap().to_string();
        assert!(error.contains("MOONCLIP_S3_ACCESS_KEY"), "{error}");

        // A role that answers.
        let (address, _) = serve(|_, _, _| (200, r#"{"AccessKeyId":"R","SecretAccessKey":"S","Token":"T"}"#.into()));
        let role = signing_with(None, None, None, env(&[("AWS_CONTAINER_CREDENTIALS_FULL_URI", &address)])).unwrap();
        assert_eq!(keys(role), "role");

        // A role that does not: both reasons in one error.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let gone = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let error = signing_with(None, None, None, env(&[("AWS_EC2_METADATA_SERVICE_ENDPOINT", &gone)]))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("MOONCLIP_S3_ACCESS_KEY") && error.contains("no role either"), "{error}");
    }

    #[test]
    fn a_machine_without_a_metadata_service_fails_within_seconds() {
        // A port nothing listens on: refused at once, as off EC2 the
        // link-local address times out within METADATA_TIMEOUT.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let role = RoleCredentials::new(RoleSource::Instance { endpoint: address });
        let error = role.get().unwrap_err().to_string();
        assert!(error.contains("did not answer"), "{error}");
    }
}
