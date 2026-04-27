/// Representation of team rating.
#[nutype::nutype(
    validate(greater = 0),
    derive(Debug, Display, Clone, Copy, Serialize, Deserialize)
)]
pub struct Rating(u16);

impl Rating {
    /// Return the rating as `f32` for probability calculations.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from(unsafe { std::mem::transmute::<Self, u16>(self) })
    }
}
